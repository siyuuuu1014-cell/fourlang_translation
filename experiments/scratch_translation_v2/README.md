# 四语随机初始化 Transformer v2（完全隔离）

这是 `scratch_translation` v1 的后续实验，不是主线 NLLB/student 模型的延续训练。

## 隔离保证

- 代码、配置、检查点、日志和评测结果全部位于 `experiments/scratch_translation_v2/`。
- 唯一共享内容是 v1 已冻结的 `prepared` 数据和 SentencePiece 词表，均以只读方式使用并逐文件校验 SHA-256。
- v2 从随机权重开始；训练代码明确记录 `pretrained_weights=false` 和 `v1_model_weights_used=false`。
- v1 解码诊断只读取 v1 模型，新增结果仍写到 v2；不会覆盖 v1 或主线任何文件。
- 输出路径有硬边界，尝试写出 v2 `artifacts/` 会立即失败。

## 相比 v1 的改动

- 模型约 5,300 万参数：`d_model=512`、6 层编码器、6 层解码器、8 头、FFN 2048、dropout 0.1。
- 训练共 60,000 次参数更新：前 50,000 步为 human+KD，后 10,000 步仅用 human 数据低学习率收尾。
- KD 占比按方向设置：`zh↔uz`、`uz↔ru` 为 50%，`zh↔ru` 为 40%，其余方向为 30%。每个采样块仍保持 12 个方向各 20,000 条，改变的是块内 human/KD 组成。
- 选择最佳检查点使用纯字符 chrF（`word_order=0`）；最终同时报告 BLEU、chrF、chrF++。
- 最终采用 beam=5；另有独立命令在 v1 验证集上公平选择 greedy/beam，再用同一种解码方式重评 v1 A/B。

## 服务器同步（在本机 PowerShell 执行）

请将下面的 `<服务器地址>` 和 SSH 端口按你的 AutoDL 实例替换。只同步新目录，不触碰服务器上的 v1 和主线：

```powershell
scp -P <端口> -r .\experiments\scratch_translation_v2 root@<服务器地址>:/root/autodl-tmp/fourlang_translation/experiments/
```

若当前目录已经位于项目根目录，可在服务器先做静态检查：

```bash
cd /root/autodl-tmp/fourlang_translation
/root/autodl-tmp/venvs/small100_student/bin/python -m compileall -q experiments/scratch_translation_v2
/root/autodl-tmp/venvs/small100_student/bin/python experiments/scratch_translation_v2/run.py status
```

## 推荐执行顺序

先做一个 20 步冒烟测试；它会产生正式 v2 的可续训检查点：

```bash
cd /root/autodl-tmp/fourlang_translation
env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/autodl-tmp/venvs/small100_student/bin/python -u \
experiments/scratch_translation_v2/run.py train --device cuda --stop-after-steps 20
```

确认输出正常后，用完全相同的命令去掉 `--stop-after-steps 20`。它会自动从第 20 步继续，一次跑完两个阶段；不需要在 50,000 步手动重启：

```bash
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/autodl-tmp/venvs/small100_student/bin/python -u \
experiments/scratch_translation_v2/run.py train --device cuda \
> scratch_v2_train.log 2>&1 &
tail -f scratch_v2_train.log
```

意外断线、关终端或进程中断后，重复同一条 `train` 命令即可从最近一次完整检查点恢复。不能改配置、代码或 suite 后混接旧检查点。

训练完整结束后执行最终测试集评测：

```bash
env PYTHONUNBUFFERED=1 /root/autodl-tmp/venvs/small100_student/bin/python -u \
experiments/scratch_translation_v2/run.py evaluate --device cuda
python -m json.tool experiments/scratch_translation_v2/artifacts/fourlang_scratch_v2/run/evaluation/beam_metrics.json
```

可选：不重训 v1，只比较其 greedy 与 beam，并保存预测样例：

```bash
nohup env PYTHONUNBUFFERED=1 \
/root/autodl-tmp/venvs/small100_student/bin/python -u \
experiments/scratch_translation_v2/run.py diagnose-v1 --device cuda \
> scratch_v2_v1_decode.log 2>&1 &
```

状态查看：

```bash
/root/autodl-tmp/venvs/small100_student/bin/python experiments/scratch_translation_v2/run.py status
```

若以后修改任何训练配置，应使用新的 suite，例如 `--suite fourlang_scratch_v2b`，避免把不同实验接到同一检查点上。
