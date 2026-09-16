"""Export the four-language M2M100 model to int8 ONNX for mobile deployment.

Full pipeline (engineered, reproducible):

    1. export fp32 ONNX (encoder + decoder + decoder_with_past + decoder_merged)
    2. int8-quantize the merged decoder (recursive into If subgraphs, weight-dedup)
    3. int8-quantize the encoder (plain dynamic quantization)
    4. package encoder + merged decoder + tokenizer into a self-contained bundle
    5. verify translations via ORTModelForSeq2SeqLM

Why a custom quantizer for the merged decoder: onnxruntime's ``quantize_dynamic``
does not recurse into ``If`` subgraphs, so it leaves the merged decoder in fp32.
This module quantizes every ``MatMul``/``Gather`` weight in the main graph and the
``then_branch``/``else_branch`` subgraphs, sharing one int8 copy per unique weight.

Usage:
    python scripts/onnx_export/quantize_fourlang.py \
        --model results/student/fourlang_m2m100/m2m100_targeted_from_human_v1/best_model/shared \
        --output onnx_export/m2m100_fourlang_mobile
"""

from __future__ import annotations

import argparse
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


def export_fp32(model_path: str, work_dir: Path) -> Path:
    """Export the model to fp32 ONNX via optimum-cli."""
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
    """Symmetric per-tensor int8 quantization of one weight (deduplicated).

    Returns ``(int8_name, scale_name, zp_name)``; registers the int8/scale/zero-
    point initializers once per unique weight.
    """
    if name in quant_cache:
        return quant_cache[name]
    import onnx  # local import keeps module import cheap
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
    import onnx
    from onnx import helper, numpy_helper

    dq_nodes: list[Any] = []
    new_nodes: list[Any] = []

    for node in graph.node:
        weight_name: str | None = None
        weight_index = 1
        if node.op_type == "MatMul" and len(node.input) >= 2:
            weight_name = node.input[1]
            weight_index = 1
        elif node.op_type == "Gather" and len(node.input) >= 1:
            weight_name = node.input[0]
            weight_index = 0

        if weight_name and weight_name in init_map:
            arr = numpy_helper.to_array(init_map[weight_name])
            if arr.dtype == np.float32:
                i8, s, z = quantize_weight(
                    weight_name,
                    arr,
                    new_inits=new_inits,
                    quant_cache=quant_cache,
                    quantized_names=quantized_names,
                )
                node_counter[0] += 1
                dq_name = f"{weight_name}_dq_{node_counter[0]}"
                dq_nodes.append(
                    helper.make_node(
                        "DequantizeLinear",
                        [i8, s, z],
                        [dq_name],
                        name=f"DQ_{node_counter[0]}",
                    )
                )
                inputs = list(node.input)
                inputs[weight_index] = dq_name
                new_nodes.append(helper.make_node(node.op_type, inputs, node.output, name=node.name))
                continue

        if node.op_type == "If":
            for attr in node.attribute:
                if attr.name in ("then_branch", "else_branch"):
                    quantize_graph_recursive(
                        attr.g,
                        init_map=init_map,
                        new_inits=new_inits,
                        quant_cache=quant_cache,
                        quantized_names=quantized_names,
                        node_counter=node_counter,
                    )
            new_nodes.append(node)
            continue

        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(dq_nodes + new_nodes)


def quantize_merged_decoder(src: Path, dst: Path) -> int:
    """Recursively int8-quantize the merged decoder; returns unique-weight count."""
    import onnx

    model = onnx.load(str(src))
    init_map = {i.name: i for i in model.graph.initializer}
    new_inits: list[Any] = []
    quant_cache: dict[str, tuple[str, str, str]] = {}
    quantized_names: set[str] = set()
    node_counter = [0]

    quantize_graph_recursive(
        model.graph,
        init_map=init_map,
        new_inits=new_inits,
        quant_cache=quant_cache,
        quantized_names=quantized_names,
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
    """Plain int8 dynamic quantization for the encoder (no If subgraphs)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def package(fp32_dir: Path, out_dir: Path, source_checkpoint: str) -> None:
    """Assemble the self-contained mobile bundle and write MANIFEST.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    quantize_encoder(fp32_dir / "encoder_model.onnx", out_dir / "encoder_model.onnx")
    n = quantize_merged_decoder(
        fp32_dir / "decoder_model_merged.onnx", out_dir / "decoder_model_merged.onnx"
    )

    for name in [
        "config.json",
        "generation_config.json",
        "sentencepiece.bpe.model",
        "vocab.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]:
        src = fp32_dir / name
        if src.exists():
            shutil.copy(src, out_dir / name)

    manifest = {
        "name": "m2m100_fourlang_mobile",
        "quantization": "int8 dynamic (recursive MatMul/Gather, per-tensor symmetric)",
        "base_model": "facebook/m2m100_418M",
        "base_license": "MIT",
        "source_checkpoint": source_checkpoint,
        "languages": {k: {"token_id": v} for k, v in LANG_TOKEN_IDS.items()},
        "special_tokens": SPECIAL_TOKENS,
        "layout": "2-file merged decoder (KV cache retained)",
        "quantized_weights": n,
        "runtime_note": "onnxruntime must use ORT_ENABLE_BASIC graph optimization",
        "files": {
            f.name: {"size_bytes": f.stat().st_size}
            for f in out_dir.iterdir()
            if f.is_file()
        },
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"packaged {len(manifest['files'])} files into {out_dir}")


def verify(model_dir: Path) -> None:
    """Smoke-test the int8 ONNX bundle against a few directions."""
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    model = ORTModelForSeq2SeqLM.from_pretrained(
        str(model_dir), use_cache=True, session_options=session_options
    )

    for src, tgt, text in [
        ("zh", "uz", "今天天气怎么样"),
        ("zh", "en", "我爱学习。"),
        ("en", "ru", "Hello, world."),
    ]:
        tokenizer.src_lang = src
        inputs = tokenizer(text, return_tensors="pt")
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(tgt),
            max_new_tokens=64,
            num_beams=2,
            early_stopping=False,
        )
        print(f"{src}-{tgt}: {tokenizer.batch_decode(generated, skip_special_tokens=True)[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the HF checkpoint.")
    parser.add_argument("--output", required=True, help="Output bundle directory.")
    parser.add_argument("--work-dir", default="onnx_export/_work", help="Scratch dir for fp32.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing fp32 dir.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the inference check.")
    args = parser.parse_args()

    work_dir = PROJECT_ROOT / args.work_dir
    fp32_dir = PROJECT_ROOT / args.work_dir / "fp32"
    if args.skip_export and fp32_dir.exists():
        print(f"reusing {fp32_dir}")
    else:
        print("exporting fp32 ...")
        export_fp32(args.model, work_dir)

    out_dir = PROJECT_ROOT / args.output
    package(fp32_dir, out_dir, args.model)

    if not args.skip_verify:
        print("verifying ...")
        verify(out_dir)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
