# 四语模型（m2m100）移动端 ONNX 集成包

一个可迁移的 int8 量化翻译模型包，覆盖 **中文 / 英文 / 俄文 / 乌兹别克语** 共 12 个方向，
用单个模型（M2M100-418M，MIT 许可）完成，可直接接入任意 App（Android / iOS / C++）。

> 当前推荐版本是 **v2（merged 2 文件，约 734MB）**，比 v1（3 文件 1.2GB）更小、接线更简单、KV cache 保留。

## 文件清单（v2，推荐）

| 文件 | 大小 | 用途 |
|---|---|---|
| `encoder_model.onnx` | 287 MB | 编码源句一次，输出 `last_hidden_state` |
| `decoder_model_merged.onnx` | 474 MB | **合并解码器**（内部 `If` 节点自动处理第一步/后续步），带 KV cache |
| `sentencepiece.bpe.model` | 2.4 MB | SentencePiece BPE 分词器 |
| `vocab.json` / `vocab.txt` | 3.7 / 2.3 MB | 词表（BPE 合并规则在 `.model` 里） |

总量约 **734 MB**（int8，含 KV cache）。

> v1（3 文件：encoder + decoder + decoder_with_past，1.2GB）仍可用，但 decoder 权重在磁盘和内存里各存两份；
> v2 的 merged decoder 把两份合成一份（权重共享），因此更小、运行时内存也更省（约 1GB vs 1.5-2GB）。

## 依赖

只依赖 **onnxruntime-mobile**（`com.microsoft.onnxruntime:onnxruntime-mobile` for Android，
或 `onnxruntime-objc` / C++ for iOS），以及一个 SentencePiece 实现（或直接用 `vocab.json`
自己做 tokenize）。

> ⚠️ **重要：图优化级别必须用 BASIC**。onnxruntime 1.30 的 EXTENDED 优化里
> `TransposeDQWeightsForMatMulNBits` 对"子图内 DQ 节点引用主图 int8 权重"这种跨图引用有 bug，
> 会报 `Missing required weight`。加载 merged decoder 时设
> `graph_optimization_level = ORT_ENABLE_BASIC`（已实测加载和推理正常）。

## 语言 token id（关键常量）

M2M100 用 `forced_bos_token_id` 指定目标语言，源语言用 `__<lang>__` 前缀 token：

| 语言 | token_id |
|---|---|
| en | 128022 |
| zh | 128102 |
| uz | 128096 |
| ru | 128077 |

特殊 token：`bos=0` `eos=2` `pad=1`。

## 推理流程（解码循环）

### 1. 编码（一次）

```
input_ids = tokenize(src_text) + src_lang 前缀   # 前缀 token = __<src>__ = lang_token_id[src]
attention_mask = [1]*len(input_ids)
last_hidden_state = encoder.run(input_ids, attention_mask)
```

### 2. 解码（自回归）

第一步（用 `decoder_model.onnx`）：

```
decoder_input = [eos] + [tgt_lang token]        # M2M100: decoder_start_token_id=2, 再拼 forced_bos
logits, present = decoder.run(
    encoder_attention_mask = attention_mask,
    input_ids = decoder_input,                   # [1, seq]
    encoder_hidden_states = last_hidden_state,
)
next_token = argmax(logits[:, -1, :])            # greedy；beam 见下
```

后续步（用 `decoder_with_past_model.onnx`，喂回 `present`）：

```
logits, present = decoder_with_past.run(
    encoder_attention_mask = attention_mask,
    input_ids = [[next_token]],
    past_key_values.0..11.decoder/encoder.key/value = present,   # 上一步输出
)
```

循环直到 `next_token == eos(2)` 或达到 `max_new_tokens`。

### 3. Beam search（可选，质量更好）

- `num_beams = k`：每一步保留 top-k 个序列，每个 beam 各跑一次 `decoder_with_past`，
  用累积 log-概率排序。对翻译任务建议 `k=2~5`（`config.json` 里 `num_beams` 默认 5）。

## 集成到 App 的最小步骤

1. 把 3 个 `.onnx` + `sentencepiece.bpe.model`（或 `vocab.json`）打进 App 资源。
2. 用 `onnxruntime-mobile` 创建 3 个 `InferenceSession`（encoder / decoder / decoder_with_past）。
3. 用 SentencePiece 分词器做 encode/decode（或按 `vocab.json` 自实现 BPE）。
4. 按上面的解码循环实现 greedy/beam 生成。
5. 翻译一个方向 = `src_lang` 前缀 token + `forced_bos_token_id`（上表）。

## 许可

- 基座 `facebook/m2m100_418M` = **MIT**，可商用。
- 微调权重源自本项目 `m2m100_targeted_from_human_v1` 实验，训练数据许可需自行核对
  （见项目内 `docs/PIPELINE_RUNBOOK.md` 的「商业许可证门禁」）。
- `sentencepiece` 模型为 Apache-2.0。

## 复现

本包由以下命令生成（服务器 `small100_student` venv，已装 `optimum[onnxruntime]`）：

```bash
PY=/root/autodl-tmp/venvs/small100_student/bin/python
MODEL=results/student/fourlang_m2m100/m2m100_targeted_from_human_v1/best_model/shared
# 1. fp32 导出
"$PY" -m optimum.commands.optimum_cli export onnx -m "$MODEL" --task seq2seq-lm-with-past onnx_export/m2m100_fourlang_fp32
# 2. int8 动态量化（onnxruntime quantize_dynamic, QInt8）
#    见 onnx_export/ 下的 quantize 脚本；量化 encoder/decoder/decoder_with_past 三个文件
```
