from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .common import PROJECT_ROOT, load_config, pair_info
except ImportError:
    from common import PROJECT_ROOT, load_config, pair_info


def parse_result(text: str) -> tuple[bool, float, str]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return False, 0.0, "unparseable"
    try:
        payload = json.loads(match.group(0))
        score = float(payload["score"])
        reason = str(payload.get("reason", ""))[:500]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False, 0.0, "unparseable"
    return 0.0 <= score <= 1.0, min(1.0, max(0.0, score)), reason


def prompt(mode: str, src_lang: str, tgt_lang: str, source: str, target: str) -> str:
    origin = "human parallel corpus" if mode == "human" else "translation teacher"
    return f"""You are a strict bilingual translation quality judge.
The candidate comes from a {origin}. Evaluate semantic equivalence, omissions, additions,
language correctness, fluency, names, numbers and negation. Do not rewrite either sentence.
Source language: {src_lang}
Target language: {tgt_lang}
Source: {source}
Candidate translation: {target}
Return JSON only: {{"score": number from 0 to 1, "reason": "short reason"}}"""


def output_paths(config: dict, mode: str, calibration: bool) -> tuple[Path, Path]:
    pair, _, _, _ = pair_info(config)
    base = PROJECT_ROOT / "data" / "pipeline_v2" / pair
    if mode == "human":
        source = base / "rule_pass.parquet"
        output = base / ("judge_calibration.parquet" if calibration else "human_judged.parquet")
    else:
        source = base / "teacher_generated.parquet"
        output = base / ("teacher_judge_calibration.parquet" if calibration else "teacher_judged.parquet")
    return source, output


def judge_id(row: dict) -> str:
    return f"{row['pair_id']}:{row.get('src_lang', row.get('source_lang'))}:{row.get('tgt_lang', row.get('target_lang'))}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven Qwen translation Judge.")
    parser.add_argument("mode", choices=("human", "teacher"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_path, output_path = output_paths(config, args.mode, args.calibration)
    frame = pd.read_parquet(source_path)
    if args.calibration:
        count = min(int(config["judge"]["calibration_pairs"]), len(frame))
        frame = frame.sample(n=count, random_state=int(config["direction"]["seed"])).sort_values("pair_id")
    if output_path.exists() and not args.overwrite:
        existing = pd.read_parquet(output_path)
        if "judge_id" not in existing:
            existing["judge_id"] = [judge_id(row) for row in existing.to_dict("records")]
        completed = set(existing.loc[existing["judge_parse_ok"], "judge_id"].astype(str))
        frame["judge_id"] = [judge_id(row) for row in frame.to_dict("records")]
        pending = frame[~frame["judge_id"].astype(str).isin(completed)]
        result = existing.to_dict("records")
    else:
        pending = frame
        result = []

    model_path = str(config["judge"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    max_new_tokens = int(config["judge"]["max_new_tokens"])
    for index, row in enumerate(pending.itertuples(index=False), 1):
        source = str(row.src_text if hasattr(row, "src_text") else row.source_text)
        target = str(row.teacher_text if args.mode == "teacher" else row.target_text)
        src_lang = str(row.src_lang if hasattr(row, "src_lang") else row.source_lang)
        tgt_lang = str(row.tgt_lang if hasattr(row, "tgt_lang") else row.target_lang)
        message = prompt(args.mode, src_lang, tgt_lang, source, target)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        answer = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parse_ok, score, reason = parse_result(answer)
        record = row._asdict()
        record["judge_id"] = judge_id(record)
        record.update({"judge_parse_ok": parse_ok, "judge_score": score, "judge_reason": reason, "judge_raw": answer[:2000]})
        result.append(record)
        if index % 100 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result).to_parquet(output_path, index=False)
            print(f"judged {index}/{len(pending)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).drop_duplicates("judge_id", keep="last").to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
