from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config_utils import load_config, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate one sentence with M2M100")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--src-lang", required=True, choices=["zh", "en", "ru", "uz"])
    parser.add_argument("--tgt-lang", required=True, choices=["zh", "en", "ru", "uz"])
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.src_lang == args.tgt_lang:
        raise ValueError("Source and target languages must be different")

    cfg = load_config(args.config)
    model_name = cfg["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, project_path(args.adapter))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    tokenizer.src_lang = args.src_lang
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(args.tgt_lang),
            max_new_tokens=128,
            num_beams=1,
        )
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
