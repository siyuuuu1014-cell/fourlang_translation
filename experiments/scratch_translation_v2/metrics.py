from __future__ import annotations

import math
import re
import unicodedata

import torch
from sacrebleu.metrics import BLEU, CHRF
from torch.nn import functional as F

from .common import DIRECTIONS
from .data import collate


def translation_loss(logits, labels, pad_id, smoothing=0.0):
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=pad_id, label_smoothing=smoothing, reduction="sum")


def clean(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def wrong_script(language, text):
    cjk = bool(re.search(r"[\u3400-\u9fff]", text))
    cyrillic = bool(re.search(r"[\u0400-\u052f]", text))
    if language == "zh":
        return bool(text) and not cjk
    if language == "ru":
        return bool(text) and not cyrillic
    if language in ("en", "uz"):
        return cjk or cyrillic
    return False


def repeated_trigram(text):
    units = list(text) if " " not in text else text.split()
    triples = [tuple(units[i:i + 3]) for i in range(max(0, len(units) - 2))]
    return bool(triples) and len(set(triples)) < len(triples)


@torch.no_grad()
def score_model(model, vocab, rows, config, device, decoding="greedy", amp=False,
                keep_samples=False):
    model.eval()
    result, saved_samples = {}, []
    batch_size = (config["evaluation"]["validation_batch_size"] if decoding == "greedy"
                  else config["evaluation"]["final_batch_size"])
    for direction in DIRECTIONS:
        part = [row for row in rows if row["direction"] == direction]
        predictions, references, loss_sum, tokens = [], [], 0.0, 0
        for start in range(0, len(part), batch_size):
            batch = part[start:start + batch_size]
            source, decoder, labels = collate(batch, vocab.pad, device)
            context = (torch.autocast(device_type="cuda", dtype=torch.float16)
                       if amp else torch.no_grad())
            with context:
                loss_sum += float(translation_loss(model(source, decoder), labels, vocab.pad))
                if decoding == "beam":
                    output = model.generate_beam(
                        source, vocab.bos, vocab.eos, config["evaluation"]["max_new_tokens"],
                        config["evaluation"]["beam_size"], config["evaluation"]["length_penalty"],
                        list(vocab.languages.values()))
                else:
                    output = model.generate_greedy(
                        source, vocab.bos, vocab.eos, config["evaluation"]["max_new_tokens"],
                        list(vocab.languages.values()))
            tokens += int(labels.ne(vocab.pad).sum())
            predictions.extend(clean(vocab.decode(sequence)) for sequence in output.cpu().tolist())
            references.extend(row["tgt_text"] for row in batch)
        target = direction.split("-")[1]
        bleu = BLEU(tokenize="zh" if target == "zh" else "13a")
        chrf = CHRF(word_order=0)
        chrfpp = CHRF(word_order=2)
        length_ratios = [len(prediction) / max(1, len(reference))
                         for prediction, reference in zip(predictions, references, strict=True)]
        result[direction] = {
            "bleu": float(bleu.corpus_score(predictions, [references]).score),
            "chrf": float(chrf.corpus_score(predictions, [references]).score),
            "chrf2": float(chrfpp.corpus_score(predictions, [references]).score),
            "loss": loss_sum / tokens, "samples": len(part),
            "empty_rate": sum(not text for text in predictions) / len(predictions),
            "wrong_script_rate": sum(wrong_script(target, text) for text in predictions) / len(predictions),
            "repeated_trigram_rate": sum(repeated_trigram(text) for text in predictions) / len(predictions),
            "mean_length_ratio": sum(length_ratios) / len(length_ratios),
        }
        if keep_samples:
            limit = config["evaluation"]["diagnostic_samples_per_direction"]
            saved_samples.extend({"direction": direction, "source": row["src_text"],
                                  "reference": reference, "prediction": prediction}
                                 for row, reference, prediction in
                                 zip(part[:limit], references[:limit], predictions[:limit], strict=True))
        print(f"{decoding} {direction}: BLEU={result[direction]['bleu']:.3f} "
              f"chrF={result[direction]['chrf']:.3f}", flush=True)
    macro = {key: sum(value[key] for value in result.values()) / len(result)
             for key in ("bleu", "chrf", "chrf2", "loss")}
    if not all(math.isfinite(value) for value in macro.values()):
        raise RuntimeError("Non-finite evaluation metric.")
    return {"directions": result, "macro": macro, "decoding": decoding,
            "samples": saved_samples if keep_samples else None}
