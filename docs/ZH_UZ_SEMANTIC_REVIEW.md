# 中乌语义复审（不训练、不导出训练集）

输入为完整 cleaning preview 目录，不是上传的抽样 packet。
审核全部隔离候选、待复核记录，以及暂留候选每方向固定随机抽检 200 条。
按方向及原文/译文精确去重，保留所有 audit_id、权重和出现次数映射。
Qwen 不会看到旧标签、来源或规则提示。输出是模型初筛，不是人工质量认证。

## 服务器运行

同步代码后，在项目根目录执行：

```bash
cd /root/autodl-tmp/fourlang_translation
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/autodl-tmp/venvs/qwen3_judge/bin/python \
  scripts/pipeline_v3/review_zh_uz_semantics.py \
  --model /root/autodl-tmp/models/Qwen3-8B \
  > fourlang_zh_uz_semantic_review_v1.log 2>&1 &
echo $!
tail -n 30 -f fourlang_zh_uz_semantic_review_v1.log
```

可先添加 `--prepare-only` 做 CPU 输入检查和队列准备，然后去掉该参数执行。
默认 batch_size=32，每批原子保存；中断后执行相同命令续跑，最多重做尚未保存的一批。
从旧 batch=8 的运行迁移时，先停止旧进程，再指定新 `--output`，并添加
`--resume-from reports/diagnostics/fourlang/zh_uz_semantic_review_v1`。
脚本校验旧块、模型及题目一致后复用已完成结果，不改写旧目录。每次续跑保留同一 resume-from 参数，旧目录不要恢复运行或删除。
如 32 显存不足，可用新目录和较小 batch，再从该次运行目录迁移已保存结果；会保留已有原始结果。
Ctrl+C 退出 tail 只停止看日志。没有自动重启、关机或训练。
首次及续跑需校验本地模型文件哈希，可能暂时没有逐批进度。
更换模型、代码、参数或输入必须换一个新的 `--output`，不覆盖原结果。
OOM 时降低 `--batch-size` 并使用新输出目录；旧目录保留。
超长输入不截断，标为 UNCERTAIN；无效 JSON 也转入待人工复核。

## 结果

```bash
python -m json.tool reports/diagnostics/fourlang/zh_uz_semantic_review_v1/summary.json
```

下载 `reports/diagnostics/fourlang/zh_uz_semantic_review_v1/semantic_review_packet.json` 查看抽样。
完整结果在同目录 `semantic_results.jsonl`；所有类别均保留，不据 PASS 自动导出训练数据。
包内每类最多 20 条，不是总体错误率估计；不能据此宣称已全量人工审核。
待确认隔离项即使本次模型判 PASS 也不自动解除隔离候选状态。
完成复核并明确筛选政策后，才另行生成清洗版及短程对照实验。

## 二次裁决预览

首轮完整运行结束后，用下面的一条命令恢复可修复的 JSON 解析失败，并对首轮
FAIL、MINOR、仍无法解析项及每方向固定 200 条 PASS 做盲二次复审：

```bash
cd /root/autodl-tmp/fourlang_translation
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/autodl-tmp/venvs/qwen3_judge/bin/python \
  scripts/pipeline_v3/adjudicate_zh_uz_semantics.py \
  --model /root/autodl-tmp/models/Qwen3-8B \
  > fourlang_zh_uz_adjudication_v1.log 2>&1 &
echo $!
tail -n 30 -f fourlang_zh_uz_adjudication_v1.log
```

同一命令可断点续跑；每批 32 条原子保存。二次提示不包含首轮标签或理由，并要求
错误类型、原文精确证据、译文精确证据和置信度。只有“首轮 FAIL + 二次 FAIL +
HIGH + 证据字段有效”才进入新增隔离候选，仍不会自动删除或改写训练数据。原登记的
隔离项保持候选状态；其他分歧进入人工复核。

结果查看：

```bash
python -m json.tool reports/diagnostics/fourlang/zh_uz_semantic_adjudication_v1/summary.json
```

供人工查看的文件为同目录 `adjudication_packet.json`；完整保守预览为
`conservative_preview.jsonl`。二次复审仍由同一模型完成，不能代替母语人工确认。

### 修复严格证据校验并仅补审未解决项

若 v1 摘要出现大量 parse failures，运行修正版。它会验证 v1 的清单、结果、分块和
模型指纹，离线接受 PASS 多填证据以及仅标点/空格不同的证据；只补审仍未解决项与
此前漏审的登记候选。不会重跑全部 1,885 条：

```bash
cd /root/autodl-tmp/fourlang_translation
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/autodl-tmp/venvs/qwen3_judge/bin/python \
  scripts/pipeline_v3/finalize_zh_uz_adjudication.py \
  --model /root/autodl-tmp/models/Qwen3-8B \
  > fourlang_zh_uz_adjudication_v2.log 2>&1 &
echo $!
tail -n 30 -f fourlang_zh_uz_adjudication_v2.log
```

输出在 `reports/diagnostics/fourlang/zh_uz_semantic_adjudication_v2`。查看：

```bash
python -m json.tool reports/diagnostics/fourlang/zh_uz_semantic_adjudication_v2/summary.json
```

供最终确认的文件是 `final_review_packet.json`。三个已发现的术语/格式误判被配置为
只进入专项复核，不能因同模型双重 FAIL 自动升级为隔离候选。所有类别仍是预览，
`training_action_applied` 始终为 false。
