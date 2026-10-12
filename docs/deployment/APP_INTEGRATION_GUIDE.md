# FourLang App 对接文档

四语（中文 / 英语 / 俄语 / 乌兹别克语）翻译模型的接入说明。模型已压缩量化为 ONNX，
可接入任意 App（Android / iOS / 桌面 / Web / 服务端）。

---

## 1. 模型清单

| model_id | 覆盖方向 | 大小 | 基座 / 许可 |
|---|---|---|---|
| `m2m100_fourlang` | 12 个方向（中英俄乌两两互译） | 734MB | m2m100_418M / MIT |
| `zh_uz` | zh-uz, uz-zh | 587MB | SMaLL-100 / MIT |
| `en_zh` | en-zh, zh-en | 587MB | SMaLL-100 / MIT |
| `en_uz` | en-uz, uz-en | 587MB | SMaLL-100 / MIT |
| `en_ru` | en-ru, ru-en | 587MB | SMaLL-100 / MIT |
| `zh_ru` | zh-ru, ru-zh | 587MB | SMaLL-100 / MIT |
| `uz_ru` | uz-ru, ru-uz | 587MB | SMaLL-100 / MIT |

- 只要 12 方向全覆盖 → 下载**一个** `m2m100_fourlang`（734MB）即可。
- 只做少数语言对、要更高质量 → 下载对应的 pair 模型。

---

## 2. 接入方式 A：在线翻译（最简单）

模型在服务端，App 只发 HTTP 请求，不装模型。

```http
POST http://<server>:8000/translate
Content-Type: application/json

{"source_lang":"zh","target_lang":"uz","text":"今天天气怎么样"}
```

响应：

```json
{
  "direction": "zh-uz",
  "source_language": "zh",
  "target_language": "uz",
  "model_name": "zh_uz_flores_relaxed_8k_ep3",
  "text": "今天天气怎么样",
  "translation": "Bugun ob-havo qanday",
  "latency_ms": 3582.4,
  "device": "cuda"
}
```

`source_lang` / `target_lang` 取 `zh` / `en` / `ru` / `uz`（不能相同）。

---

## 3. 接入方式 B：离线翻译（下载模型，断网可用）

### 3.1 拿模型清单

```http
GET http://<server>:8000/models/manifest
```

响应（节选）：

```json
{
  "models": [
    {
      "model_id": "m2m100_fourlang",
      "family": "m2m100",
      "version": "2.0.0",
      "source_sha256": "65d0d7b9...",
      "size_bytes": 769622935,
      "directions": ["en-ru","en-uz","en-zh","ru-en","ru-uz","ru-zh","uz-en","uz-ru","uz-zh","zh-en","zh-ru","zh-uz"],
      "download_url": "/download/m2m100_fourlang"
    },
    { "model_id": "zh_uz", "...": "..." }
  ]
}
```

### 3.2 下载模型包

```http
GET http://<server>:8000/download/{model_id}
```

返回单个 tar（支持断点续传，`accept-ranges: bytes`）。tar 内的目录结构：

```
m2m100_fourlang_mobile/
  encoder_model.onnx            # 编码源句一次
  decoder_model_merged.onnx     # 自回归解码（内部 If 处理 KV cache）
  sentencepiece.bpe.model       # BPE 分词器（合并规则在里面）
  vocab.json / vocab.txt        # 词表（辅助）
  config.json / tokenizer_config.json / special_tokens_map.json / added_tokens.json
  README.md / MANIFEST.json
```

### 3.3 本地推理

用 **onnxruntime**（对应语言的绑定）加载两个 ONNX + 分词器，自己写解码循环。见第 5 节。

### 3.4 版本更新

App 启动时调一次 `/models/manifest`，对比本地已下载包的 `source_sha256`：

- 一致 → 用本地包，离线翻译。
- 不一致 → 重新 `GET /download/{model_id}`，覆盖本地。

---

## 4. 必须注意的约束

1. **onnxruntime 图优化级别必须 `ORT_ENABLE_BASIC`**，不能用 EXTENDED
   （1.30 的优化器对 merged decoder 的跨子图 int8 权重有 bug）。
2. **语言 token id**（写进分词/解码里）：

   | 语言 | token_id |
   |---|---|
   | en | 128022 |
   | zh | 128102 |
   | uz | 128096 |
   | ru | 128077 |

   特殊 token：`bos=0` `eos=2` `pad=1`。
3. **分词器用 `sentencepiece.bpe.model`**（BPE 合并规则在里面），不要只用 `vocab.json`。
4. 端侧内存：单实例运行 ~1GB，**限制并发**（同一时刻只跑一个翻译）。

---

## 5. 解码循环（伪代码）

模型是编码器-解码器（seq2seq）。两种方向的机制不同：

### 5.1 四语模型（m2m100）

```
# 编码
src_ids = tokenize(src_text)                  # sentencepiece
src_ids = [lang_id[src]] + src_ids + [eos]    # 源语言前缀
enc_out = encoder.run(src_ids, attention_mask)

# 解码（自回归，greedy）
dec_ids = [eos]                               # decoder 起始
循环到 eos 或 max_new_tokens:
    logits = decoder.run(dec_ids, enc_out, mask,
                         forced_bos_token_id = lang_id[tgt])   # 目标语言
    next = argmax(logits[-1])
    dec_ids.append(next)
translation = decode(dec_ids[1:])             # 去掉起始 eos
```

### 5.2 六组小模型（SMaLL-100）

```
# 编码
src_ids = tokenize(src_text)
src_ids = [lang_id[tgt]] + src_ids + [eos]    # 注意：目标语言前缀（不是源语言）
enc_out = encoder.run(src_ids, attention_mask)

# 解码（无 forced_bos，从 eos 开始）
dec_ids = [eos]
循环到 eos:
    logits = decoder.run(dec_ids, enc_out, mask)
    next = argmax(logits[-1])
    dec_ids.append(next)
translation = decode(dec_ids[1:])
```

> 区别：**m2m100 用源语言前缀 + forced_bos(目标)；small100 用目标语言前缀、无 forced_bos**。
> 每个包的 `README.md` 里有更详细的说明。

---

## 6. 服务地址

- 开发：`http://<server>:8000`
- 正式：把模型包上传到 CDN / 对象存储（如阿里云 OSS），把下载 URL 改成 OSS 地址即可，App 端把 URL 做成配置项，代码不用改。

---

## 7. 许可

- 全部模型基座均为 **MIT**（m2m100_418M / SMaLL-100），可商用。
- 训练数据许可需自行核对（见项目 `docs/PIPELINE_RUNBOOK.md`）。
- `sentencepiece` 为 Apache-2.0。
