"""Export a translation model (M2M100 or SMaLL-100) to int8 ONNX for mobile.

Reproducible, family-aware pipeline:

    1. export fp32 ONNX (encoder + decoder + decoder_with_past + decoder_merged)
    2. int8-quantize the merged decoder (recursive into If subgraphs, weight-dedup)
    3. int8-quantize the encoder (plain dynamic quantization)
    4. package encoder + merged decoder + tokenizer into a self-contained bundle
    5. verify translations via a greedy decode loop

The only family difference is the tokenizer and the direction mechanism:

    - m2m100 : src_lang prefix + forced_bos (target lang)
    - small100: tgt_lang prefix (target lang), no forced_bos

Both families share the M2M100 architecture and the same language token ids
(en=128022, zh=128102, uz=128096, ru=128077), so the ONNX export/quantization is
identical.

Usage:
    python scripts/onnx_export/quantize_model.py \
        --model results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared \
        --family small100 --version 1.0.0 \
        --output onnx_export/zh_uz_mobile
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LANG_TOKEN_IDS = {"en": 128022, "zh": 128102, "uz": 128096, "ru": 128077}
SPECIAL_TOKENS = {"bos_token_id": 0, "eos_token_id": 2, "pad_token_id": 1}

FAMILY_TOKENIZER_FILES = {
    "m2m100": [
        "config.json",
        "generation_config.json",
        "sentencepiece.bpe.model",
        "vocab.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ],
    "small100": [
        "config.json",
        "generation_config.json",
        "sentencepiece.bpe.model",
        "vocab.json",
        "tokenization_small100.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ],
}


def export_fp32(model_path: str, work_dir: Path) -> Path:
    out = work_dir / "fp32"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "optimum.commands.optimum_cli",
            "export",
            "onnx",
            "-m",
            model_path,
            "--task",
            "seq2seq-lm-with-past",
            str(out),
        ],
        check=True,
    )
    return out


def quantize_weight(
    name: str,
    arr: np.ndarray,
    *,
    new_inits: list[Any],
    quant_cache: dict[str, tuple[str, str, str]],
    quantized_names: set[str],
) -> tuple[str, str, str]:
    """Symmetric per-tensor int8 quantization of one weight (deduplicated)."""
    if name in quant_cache:
        return quant_cache[name]
    from onnx import numpy_helper

    arr = arr.astype(np.float32)
    scale = float(np.max(np.abs(arr))) / 127.0
    if scale < 1e-9:
        scale = 1e-9
    w_int8 = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    int8_name, scale_name, zp_name = f"{name}_q", f"{name}_s", f"{name}_z"
    new_inits.append(numpy_helper.from_array(w_int8, name=int8_name))
    new_inits.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), name=scale_name))
    new_inits.append(numpy_helper.from_array(np.array(0, dtype=np.int8), name=zp_name))
    quant_cache[name] = (int8_name, scale_name, zp_name)
    quantized_names.add(name)
    return int8_name, scale_name, zp_name


def quantize_graph_recursive(
    graph: Any,
    *,
    init_map: dict[str, Any],
    new_inits: list[Any],
    quant_cache: dict[str, tuple[str, str, str]],
    quantized_names: set[str],
    node_counter: list[int],
) -> None:
    """Quantize MatMul/Gather weights in a graph and its If subgraphs (in place)."""
    from onnx import helper, numpy_helper

    dq_nodes: list[Any] = []
    new_nodes: list[Any] = []

    for node in graph.node:
        weight_name: str | None = None
        weight_index = 1
        if node.op_type == "MatMul" and len(node.input) >= 2:
            weight_name, weight_index = node.input[1], 1
        elif node.op_type == "Gather" and len(node.input) >= 1:
            weight_name, weight_index = node.input[0], 0

        if weight_name and weight_name in init_map:
            arr = numpy_helper.to_array(init_map[weight_name])
            if arr.dtype == np.float32:
                i8, s, z = quantize_weight(
                    weight_name, arr,
                    new_inits=new_inits, quant_cache=quant_cache,
                    quantized_names=quantized_names,
                )
                node_counter[0] += 1
                dq_name = f"{weight_name}_dq_{node_counter[0]}"
                dq_nodes.append(helper.make_node("DequantizeLinear", [i8, s, z], [dq_name], name=f"DQ_{node_counter[0]}"))
                inputs = list(node.input)
                inputs[weight_index] = dq_name
                new_nodes.append(helper.make_node(node.op_type, inputs, node.output, name=node.name))
                continue

        if node.op_type == "If":
            for attr in node.attribute:
                if attr.name in ("then_branch", "else_branch"):
                    quantize_graph_recursive(
                        attr.g, init_map=init_map, new_inits=new_inits,
                        quant_cache=quant_cache, quantized_names=quantized_names,
                        node_counter=node_counter,
                    )
            new_nodes.append(node)
            continue

        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(dq_nodes + new_nodes)


def quantize_merged_decoder(src: Path, dst: Path) -> int:
    import onnx

    model = onnx.load(str(src))
    init_map = {i.name: i for i in model.graph.initializer}
    new_inits: list[Any] = []
    quant_cache: dict[str, tuple[str, str, str]] = {}
    quantized_names: set[str] = set()
    node_counter = [0]
    quantize_graph_recursive(
        model.graph, init_map=init_map, new_inits=new_inits,
        quant_cache=quant_cache, quantized_names=quantized_names,
        node_counter=node_counter,
    )
    model.graph.initializer.extend(new_inits)
    kept = [i for i in model.graph.initializer if i.name not in quantized_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return len(quant_cache)


def quantize_encoder(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def source_sha256(model_path: str) -> str:
    weights = Path(model_path) / "model.safetensors"
    if not weights.exists():
        return "unknown"
    h = hashlib.sha256()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package(fp32_dir: Path, out_dir: Path, *, family: str, model_path: str, version: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    quantize_encoder(fp32_dir / "encoder_model.onnx", out_dir / "encoder_model.onnx")
    n = quantize_merged_decoder(fp32_dir / "decoder_model_merged.onnx", out_dir / "decoder_model_merged.onnx")

    # Tokenizer/config files come from the ORIGINAL checkpoint (the fp32 export
    # does not carry the custom tokenization_small100.py for SMaLL-100).
    src_model = Path(model_path)
    if not src_model.is_absolute():
        src_model = (PROJECT_ROOT / src_model).resolve()
    for name in FAMILY_TOKENIZER_FILES[family]:
        src = src_model / name
        if src.exists():
            shutil.copy(src, out_dir / name)

    manifest = {
        "name": Path(model_path).name or "model",
        "version": version,
        "family": family,
        "quantization": "int8 dynamic (recursive MatMul/Gather, per-tensor symmetric)",
        "base_model": "facebook/m2m100_418M" if family == "m2m100" else "alirezamsh/small100",
        "base_license": "MIT",
        "source_checkpoint": model_path,
        "source_sha256": source_sha256(model_path),
        "languages": {k: {"token_id": v} for k, v in LANG_TOKEN_IDS.items()},
        "special_tokens": SPECIAL_TOKENS,
        "direction_mechanism": (
            "src_lang prefix + forced_bos(target)" if family == "m2m100"
            else "tgt_lang prefix (target), no forced_bos"
        ),
        "layout": "2-file merged decoder (KV cache retained)",
        "quantized_weights": n,
        "runtime_note": "onnxruntime must use ORT_ENABLE_BASIC graph optimization",
        "files": {f.name: {"size_bytes": f.stat().st_size} for f in out_dir.iterdir() if f.is_file()},
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"packaged {len(manifest['files'])} files into {out_dir}")


def verify(model_dir: Path, family: str) -> None:
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from inference.loader import _load_tokenizer

    tokenizer, _kind = _load_tokenizer(str(model_dir), None)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    model = ORTModelForSeq2SeqLM.from_pretrained(str(model_dir), use_cache=True, session_options=so)

    def translate(src: str, tgt: str, text: str) -> str:
        if family == "small100":
            tokenizer.tgt_lang = tgt
        else:
            tokenizer.src_lang = src
        inputs = tokenizer(text, return_tensors="pt")
        gen_kwargs = {"max_new_tokens": 60, "num_beams": 2, "early_stopping": False}
        if family == "m2m100":
            gen_kwargs["forced_bos_token_id"] = tokenizer.get_lang_id(tgt)
        gen = model.generate(**inputs, **gen_kwargs)
        return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

    cases = {
        "small100": [("zh", "uz", "今天天气怎么样"), ("uz", "zh", "Salom, dunyo.")],
        "m2m100": [("zh", "uz", "今天天气怎么样"), ("en", "ru", "Hello, world.")],
    }[family]
    for src, tgt, text in cases:
        print(f"{src}-{tgt}: {translate(src, tgt, text)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the HF checkpoint.")
    parser.add_argument("--family", choices=("m2m100", "small100"), default="m2m100")
    parser.add_argument("--version", default="1.0.0", help="Package version tag.")
    parser.add_argument("--output", required=True, help="Output bundle directory.")
    parser.add_argument("--work-dir", default="onnx_export/_work", help="Scratch dir for fp32.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing fp32 dir.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the inference check.")
    args = parser.parse_args()

    work_dir = PROJECT_ROOT / args.work_dir
    fp32_dir = work_dir / "fp32"
    if args.skip_export and fp32_dir.exists():
        print(f"reusing {fp32_dir}")
    else:
        print("exporting fp32 ...")
        export_fp32(args.model, work_dir)

    out_dir = PROJECT_ROOT / args.output
    package(fp32_dir, out_dir, family=args.family, model_path=args.model, version=args.version)

    if not args.skip_verify:
        print("verifying ...")
        verify(out_dir, args.family)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
