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
默认 batch_size=8，每批原子保存；中断后执行相同命令续跑，最多重做尚未保存的一批。
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
