from __future__ import annotations
import argparse, gc, json, math, re, time
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path("/root/autodl-tmp/fourlang_translation").resolve()
EXP1 = ROOT / "results/specialists/zh_en/opus_mt_zh_en/exp1_human/best_model"
EXP2 = ROOT / "results/specialists/zh_en/opus_mt_zh_en/exp2_kd_v1"
VAL = ROOT / "data/splits/zh_en/v1/validation_pairs_v1.parquet"
FLORES = ROOT / "data/benchmark/zh_en/flores_plus_zh_en_devtest_v1.parquet"
TATOEBA = ROOT / "data/benchmark/zh_en/tatoeba_zh_en_test_v1.parquet"
OUT = ROOT / "results/specialists/zh_en/opus_mt_zh_en/final_generation_eval_v1"

@dataclass
class R:
    model_name: str
    model_path: str
    dataset: str
    rows: int
    bleu: float
    chrfpp: float
    exact_match_pct: float
    generation_seconds: float
    seconds_per_sample: float
    samples_per_second: float
    peak_gpu_memory_gib: float | None

def load_df(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(path)

def infer_cols(df):
    pairs = [
        ("source","target"), ("src","tgt"), ("src_text","tgt_text"),
        ("source_text","target_text"), ("text_zh","text_en"),
        ("zh_text","en_text"), ("zh","en"), ("sentence_zh","sentence_en"),
        ("chinese","english"),
    ]
    cols = set(df.columns)
    for a,b in pairs:
        if a in cols and b in cols:
            return a,b
    raise KeyError(f"Cannot infer columns from: {list(df.columns)}")

def normalize(df):
    a,b = infer_cols(df)
    if "direction" in df.columns:
        mask = df["direction"].astype(str).str.lower().eq("zh_en")
        if mask.any():
            df = df.loc[mask].copy()
    out = pd.DataFrame({
        "source": df[a].astype(str).str.strip(),
        "reference": df[b].astype(str).str.strip(),
    })
    out = out[
        out["source"].ne("") & out["reference"].ne("") &
        out["source"].ne("nan") & out["reference"].ne("nan")
    ].reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No usable zh_en rows.")
    return out

def is_model_dir(p):
    return p.is_dir() and (p/"config.json").exists() and (
        (p/"model.safetensors").exists()
        or (p/"pytorch_model.bin").exists()
        or any(p.glob("model-*.safetensors"))
    )

def discover_exp2(root):
    found = []
    if is_model_dir(root):
        found.append(root)
    if root.exists():
        for p in root.iterdir():
            if is_model_dir(p):
                found.append(p)
    uniq, seen = [], set()
    for p in found:
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(p); seen.add(rp)
    def key(p):
        nums = re.findall(r"\d+", p.name)
        return (0 if "epoch" in p.name.lower() else 1, int(nums[-1]) if nums else 9999, p.name)
    uniq.sort(key=key)
    return [(f"EXP2_{p.name.upper()}", p) for p in uniq]

def generate(model_path, texts, batch_size, beams, max_src, max_new):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    preds = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_src)
            enc = {k:v.to(device) for k,v in enc.items()}
            out = model.generate(
                **enc, num_beams=beams, do_sample=False,
                max_new_tokens=max_new, early_stopping=True
            )
            preds.extend(x.strip() for x in tok.batch_decode(out, skip_special_tokens=True))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
    del model, tok
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds, elapsed, peak

def metrics(preds, refs):
    bleu = BLEU(tokenize="13a", effective_order=True).corpus_score(preds, [refs]).score
    chrfpp = CHRF(char_order=6, word_order=2, beta=2).corpus_score(preds, [refs]).score
    exact = 100.0 * sum(p.strip() == r.strip() for p,r in zip(preds,refs)) / len(refs)
    return float(bleu), float(chrfpp), float(exact)

def eval_one(name, path, dsname, df, args, pred_dir):
    print("\n"+"="*100)
    print("MODEL  :", name)
    print("PATH   :", path)
    print("DATASET:", dsname, "| rows =", len(df))
    preds, elapsed, peak = generate(
        path, df["source"].tolist(), args.batch_size, args.num_beams,
        args.max_source_length, args.max_new_tokens
    )
    refs = df["reference"].tolist()
    bleu, chrfpp, exact = metrics(preds, refs)
    print(f"BLEU={bleu:.6f} | chrF++={chrfpp:.6f} | Exact={exact:.6f}%")
    print(f"time={elapsed:.3f}s | sec/sample={elapsed/len(df):.6f}")
    if peak is not None:
        print(f"peak_gpu_memory={peak:.3f} GiB")
    safe = re.sub(r"[^A-Za-z0-9_.-]+","_",name.lower())
    p = pred_dir / f"{safe}__{dsname}.parquet"
    x = df.copy()
    x["prediction"] = preds
    x["exact_match"] = x["prediction"].str.strip().eq(x["reference"].str.strip())
    x.to_parquet(p, index=False)
    return R(
        name, str(path), dsname, len(df), bleu, chrfpp, exact, elapsed,
        elapsed/len(df), len(df)/elapsed if elapsed else math.inf, peak
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-source-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("="*100)
    print("STEP 20B - ZH->EN FINAL GENERATION EVALUATION")
    print("="*100)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))

    if not EXP1.exists():
        raise FileNotFoundError(EXP1)
    if not EXP2.exists():
        raise FileNotFoundError(EXP2)

    models = [("EXP1_BASELINE", EXP1)] + discover_exp2(EXP2)
    if len(models) < 2:
        raise RuntimeError(f"No Exp2 model checkpoints found under {EXP2}")

    print("\nCandidates:")
    for n,p in models:
        print("-", n, "=>", p)

    datasets = {
        "frozen_validation": normalize(load_df(VAL)),
        "flores_devtest": normalize(load_df(FLORES)),
        "tatoeba": normalize(load_df(TATOEBA)),
    }
    if args.limit:
        datasets = {k:v.head(args.limit).copy() for k,v in datasets.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    pred_dir = OUT/"predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n,p in models:
        for dsname,df in datasets.items():
            rows.append(asdict(eval_one(n,p,dsname,df,args,pred_dir)))

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT/"zh_en_exp1_vs_exp2_metrics_v1.csv", index=False)
    (OUT/"zh_en_exp1_vs_exp2_metrics_v1.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    held = rdf[rdf["dataset"].isin(["flores_devtest","tatoeba"])].copy()
    comp = held.groupby(["model_name","model_path"], as_index=False).agg(
        heldout_mean_bleu=("bleu","mean"),
        heldout_mean_chrfpp=("chrfpp","mean"),
        heldout_mean_exact_match_pct=("exact_match_pct","mean"),
    )
    base = comp[comp["model_name"].eq("EXP1_BASELINE")].iloc[0]
    comp["delta_bleu_vs_exp1"] = comp["heldout_mean_bleu"] - float(base["heldout_mean_bleu"])
    comp["delta_chrfpp_vs_exp1"] = comp["heldout_mean_chrfpp"] - float(base["heldout_mean_chrfpp"])
    comp = comp.sort_values(
        ["heldout_mean_bleu","heldout_mean_chrfpp"], ascending=False
    ).reset_index(drop=True)
    comp.insert(0, "rank", range(1, len(comp)+1))
    comp.to_csv(OUT/"zh_en_final_checkpoint_comparison_v1.csv", index=False)

    winner = comp.iloc[0]
    decision = {
        "direction": "zh_en",
        "selection_rule": "maximize mean BLEU on FLORES devtest + Tatoeba; chrF++ tie-breaker",
        "winner": str(winner["model_name"]),
        "winner_path": str(winner["model_path"]),
        "heldout_mean_bleu": float(winner["heldout_mean_bleu"]),
        "heldout_mean_chrfpp": float(winner["heldout_mean_chrfpp"]),
        "delta_bleu_vs_exp1": float(winner["delta_bleu_vs_exp1"]),
        "delta_chrfpp_vs_exp1": float(winner["delta_chrfpp_vs_exp1"]),
    }
    (OUT/"zh_en_final_decision_v1.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n"+"="*100)
    print("FINAL COMPARISON")
    print("="*100)
    print(comp[[
        "rank","model_name","heldout_mean_bleu","heldout_mean_chrfpp",
        "delta_bleu_vs_exp1","delta_chrfpp_vs_exp1"
    ]].to_string(index=False))
    print("\nFINAL DECISION")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("\nSTATUS: ZH_EN_FINAL_GENERATION_EVAL_COMPLETE")

if __name__ == "__main__":
    main()
