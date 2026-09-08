# 独立实验：从零训练四语 Transformer

本目录不导入、不修改 `scripts/pipeline_v2`、`scripts/pipeline_v3`，也不加载 NLLB、SMALL100、M2M100 或其他预训练权重。与原实验的唯一数据连接是**只读现有 `data/` 文件**。

两组都覆盖英、中、乌、俄的 12 个翻译方向，每组训练一个模型：

| 组别 | 独立入口 | 权重起点 | 训练样本 |
| --- | --- | --- | --- |
| A：`human_only` | `human_only.py` | 随机初始化 | 人工平行数据 |
| B：`human_kd` | `human_kd.py` | 相同随机初始化 | 70% 人工 + 30% 已筛选 Teacher KD |

B **不是**接着 A 训练，也不是原 NLLB Exp2 的续训。它使用教师生成的译文作为监督标签，即序列级蒸馏；不需要教师在线运行，不读取教师 logits。这里的“人工”指原数据集中带人工参考译文的平行数据，不表示本项目重新雇人逐句翻译。

## 隔离与目录

```text
experiments/scratch_translation/
  config.toml             # 两组共同、独立于 NLLB 的实验配置
  run.py                  # prepare / status / compare
  human_only.py           # A：train / evaluate
  human_kd.py             # B：train / evaluate
  common.py data.py model.py engine.py cli.py
  tests/test_scratch.py
  artifacts/<suite>/      # 程序的全部产物，只允许写这里
    prepared/             # 新实验的数据副本、人工训练语料词表、数据指纹
    runs/human_only/      # A 独占权重、优化器、断点、日志、评估、词表副本
    runs/human_kd/        # B 独占同类产物
    comparison.json
  logs/                   # 用户通过 nohup 重定向的日志
```

所有 `artifacts/`、`logs/` 均被本目录的 `.gitignore` 排除。程序拒绝把产物路径指向原 `data/`、`models/`、`results/` 或本目录之外。两组复用同一套实现以控制变量，但没有共享可变模型状态或交叉加载权重的入口。词表用同一份**人工训练文本**生成，随后各组持有自己的校验副本。

## 默认首轮方案

- Encoder 4 层、Decoder 4 层；隐藏维度 384、6 个注意力头、FFN 1536，pre-norm、dropout 0.2。
- 自建 SentencePiece Unigram 16k 词表，四语文本等量抽样；编码器输入包含源语言和目标语言标签，输入/输出嵌入共享。
- 最大序列长度 256；词表达到 16k 时约 **2281 万参数**。这不是一个 600M 预训练模型的缩小权重副本。
- AdamW、学习率峰值 `5e-4`、2000 步 warmup 后 inverse-square-root 衰减、label smoothing 0.1、梯度裁剪 1.0；CUDA 上启用 FP16。
- 每组 30,000 次优化器更新；物理 batch 16，梯度累积 4，有效 batch 64，共 **192 万次样本抽取**，不是 192 万条独立句对。
- 每 500 步保存完整断点；每 1000 步在固定验证子集评估，以 12 方向宏平均指标选择最佳断点；固定预算跑满，不因某组提前早停而改变对照预算。
- 验证子集最多每方向 100 条，共 1200 条；完整原验证集仍全部参与防泄漏检查。最终用 FLORES devtest 每方向全量评估。

这些是可运行的 pilot 默认值，**尚未用真实数据调优，不能保证 B 优于 A，也不承诺超过已有 NLLB**。30% KD 是这个独立实验的新起点，并非沿用原 NLLB Exp2 的 60%。先看数据报告里的截断量和训练曲线，再决定下一套实验配置。不要直接套用 NLLB 的轮数或耗时估计；新模型从零学习，验证生成也会占时间。

## 数据使用与公平性

`prepare` 从配置中的六对语言的完整人工训练文件、已完成过滤的 KD 训练文件读取数据，不使用原四语聚合后的 158,000 条采样结果。兼容旧文件的 `source_text/target_text`、`direction=en_zh`、`training_origin/sample_origin` 等字段。

KD 组合文件必须提供明确的 `training_source` 或兼容来源字段，只提取来源包含 `teacher` 的记录；不猜测无来源的 KD。人工/验证数据发现 teacher、pseudo、synthetic 来源会直接拒绝。本入口依赖上游 KD 已完成审核，不重新运行 Qwen，也不擅自放宽上游准入策略。

准备时会统一中文简体、乌语拉丁文字形，去除完全重复行、与任何语言对验证集或 FLORES dev/devtest **任一侧文本重合**的训练行，以及与人工参考完全一致的 KD 行。共享源文件在准备前后做 SHA256 校验，绝不原地修改。验证和测试文本不会进入词表训练。

每个采样块对每个方向抽取 20,000 个样本，块内混洗：

- A：20,000 人工；B：14,000 人工 + 6,000 KD。
- 从完整清洗数据池按固定随机覆盖顺序轮换，跨块保留位置；先覆盖池中所有行，再循环，不长期只用一份静态子集。
- 人工不足会重复抽样，不产生新的翻译知识。B 的每条 KD 在单个采样块最多重复 3 次；若清洗后不满足配额，会报错而非静默突破限制。
- 每块 240,000 次抽样，默认训练恰好覆盖 8 块。两个模型具有相同初始权重、词表、方向配额、结构、种子和优化器更新预算。

这是“人工监督”与“人工 + KD 监督”的训练方案对照：B 减少了人工抽取、加入了 KD，而 KD 的源句覆盖也可能不同。它**不是**严格控制同一句源文、只替换译文的因果实验；等样本/等更新次数也不代表等非 padding token 数。日志记录实际目标 token 数。单一种子只适合初步比较，不能据此宣称统计显著提升。

## 服务器运行

先将整个 `experiments/scratch_translation/` 文件夹同步到服务器仓库的同名目录，保持文件结构。不要同步本地测试产物。以下命令在服务器的 Bash 中执行，均是**手动操作示例，代码交付不会自动开始训练**。

### 1. 检查运行环境

要求 Python 3.11+，PyTorch 2.4+ 以及 `requirements.txt` 中的库。优先检查现有环境，**不要在 NLLB 正在运行的环境里升级依赖**。若缺依赖，另建独立虚拟环境后再安装本目录的 requirements；本实验不需要下载预训练模型。

```bash
cd /root/autodl-tmp/fourlang_translation
/root/autodl-tmp/venvs/small100_student/bin/python -c "import torch, pandas, numpy, pyarrow, sentencepiece, sacrebleu, opencc, filelock; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

### 2. 准备并审查数据

```bash
/root/autodl-tmp/venvs/small100_student/bin/python -u \
  experiments/scratch_translation/run.py prepare

python -m json.tool \
  experiments/scratch_translation/artifacts/fourlang_scratch_v1/prepared/manifest.json
```

需要服务器上原先的完整人工/KD 文件和四语 FLORES 文件。缺文件会明确给出路径。prepare 会打印逐语言对的读取、清洗、词表和编码进度；成功后相同输入可再次运行，校验后复用。

报告含每个方向的可用样本数、保护集排除数、截断量、实际词表规模与文件指纹。先确认数据量和截断量合理，再训练。若要更改配置，请用新 suite，并对**两组**保持相同设置，例如三个入口都加 `--suite fourlang_scratch_v2`；不允许改配置后悄悄接旧断点。

### 3. 可选：短跑验证，再继续同一组

确保 GPU 没被当前 NLLB Exp2 占用。单张 V100 上应串行运行这些训练，不要同时启动两组或与 NLLB 抢显存。

```bash
/root/autodl-tmp/venvs/small100_student/bin/python -u \
  experiments/scratch_translation/human_only.py train --device cuda --stop-after-steps 20
```

这只是先走 20 步并保存，**不改变 30,000 步总预算**。之后去掉 `--stop-after-steps`，会自动从这组的第 20 步续训。首轮短跑同时用于测量显存和训练速度；评估耗时需等第一次验证才能估计。

### 4. 正式训练 A

```bash
mkdir -p experiments/scratch_translation/logs
nohup /root/autodl-tmp/venvs/small100_student/bin/python -u \
  experiments/scratch_translation/human_only.py train --device cuda \
  >> experiments/scratch_translation/logs/human_only.log 2>&1 &
echo $!
tail -f experiments/scratch_translation/logs/human_only.log
```

按 Ctrl+C 退出 `tail -f` 只退出看日志，不是杀掉 nohup 训练。`nohup` 能防终端断开，但不能防服务器重启、OOM 或人为杀进程。

### 5. A 完成后再训练 B

```bash
nohup /root/autodl-tmp/venvs/small100_student/bin/python -u \
  experiments/scratch_translation/human_kd.py train --device cuda \
  >> experiments/scratch_translation/logs/human_kd.log 2>&1 &
echo $!
tail -f experiments/scratch_translation/logs/human_kd.log
```

这两个入口各自只执行本组训练；不会自动接着启动另一组，也不会自动运行最终测试集。

### 6. 查看断点、评估、对照

```bash
/root/autodl-tmp/venvs/small100_student/bin/python \
  experiments/scratch_translation/run.py status

/root/autodl-tmp/venvs/small100_student/bin/python -u \
  experiments/scratch_translation/human_only.py evaluate --device cuda

/root/autodl-tmp/venvs/small100_student/bin/python -u \
  experiments/scratch_translation/human_kd.py evaluate --device cuda

/root/autodl-tmp/venvs/small100_student/bin/python \
  experiments/scratch_translation/run.py compare

python -m json.tool \
  experiments/scratch_translation/artifacts/fourlang_scratch_v1/comparison.json
```

`status` 只读取落盘报告，并不证明进程仍在运行。`evaluate` 要求本组预算已跑满，加载验证集选出的最佳断点，而非默认最后一个断点；评估中断后可重新执行，但评估不做逐方向断点续算。

两组统一使用贪心生成，BLEU 中文目标采用 `zh` 分词、其余为 `13a`。字段 `chrf2` 延续原项目命名，实际为 SacreBLEU **chrF++（beta=2、word_order=2）**，同时记录完整指标签名。与旧 NLLB 指标比较前要统一分词、文本规范化和解码设置，不能只对照数字。

FLORES devtest 在此前 NLLB 实验中已被查看过，因此这是复用的固定基准，不应描述成整个研究从未见过的盲测。新两组的断点选择只用本目录固定验证子集，不用测试集。

## 断点和恢复边界

- 重跑同组、同 suite 的 `train` 命令自动恢复模型、AdamW 状态、FP16 scaler、随机数状态、采样游标、训练步数和最佳断点记录。
- 断点先写临时文件，再原子提交，最后发布 `latest.json`；校验 SHA256。未提交临时文件不参与恢复。进程意外中断会从上次成功落盘处重算一小段，不保证保留中断瞬间那一批。
- 默认保留最近 3 份完整断点，加上最佳断点；更早的非最佳断点会自动删除。不能依靠这个目录恢复已经清理的旧步数。
- 同组锁防止双进程同时训练；另一组的断点即使被误复制过来也会因组身份不同而拒绝加载。
- 配置、代码、数据副本、依赖版本及 CPU/CUDA 类型均被校验。需要改变时使用新 suite，不要删除 manifest 或伪造指纹。已完成后再次运行 train 不会额外训练。
- CPU 小样本测试验证了中断恢复与连续训练的逐参数一致性；不同硬件/软件版本之间不保证位级一致。当前训练循环仅支持单进程、单 GPU。
- FP16 梯度溢出会降低 scale，重试同一批数据，不计成成功训练步；持续异常或非有限 loss 会停止，避免写入坏模型。

## 本地测试

```bash
python -m pytest experiments/scratch_translation/tests -q
```

测试使用临时小数据和小模型，在 CPU 上跑通两组训练、恢复和评估，不使用真实训练数据、不加载预训练权重、不启动长时间 GPU 训练。包括原始数据不变、跨语言对验证集隔离、词表无验证/测试/KD 文本泄漏、mask、采样、断点损坏和串组保护等检查。真实数据规模的准备与 V100 完整训练仍需服务器验证。
