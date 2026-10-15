from __future__ import annotations

import math

import torch
from torch import nn


class TranslationTransformer(nn.Module):
    """Random-init pre-norm encoder-decoder Transformer; no pretrained weights."""

    def __init__(self, vocab_size: int, config: dict, pad_id: int = 0):
        super().__init__()
        self.settings = dict(config)
        self.pad_id = pad_id
        width = config["d_model"]
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.position = nn.Embedding(config["max_length"], width)
        self.dropout = nn.Dropout(config["dropout"])
        self.transformer = nn.Transformer(
            d_model=width, nhead=config["nhead"],
            num_encoder_layers=config["encoder_layers"],
            num_decoder_layers=config["decoder_layers"],
            dim_feedforward=config["dim_feedforward"], dropout=config["dropout"],
            activation="relu", batch_first=True, norm_first=True,
        )
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=width ** -0.5)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()

    def embed(self, ids):
        if ids.size(1) > self.settings["max_length"]:
            raise ValueError("Token sequence exceeds positional capacity.")
        position = torch.arange(ids.size(1), device=ids.device)
        return self.dropout(self.embedding(ids) * math.sqrt(self.settings["d_model"])
                            + self.position(position)[None])

    def encode(self, source):
        padding = source.eq(self.pad_id)
        return self.transformer.encoder(self.embed(source), src_key_padding_mask=padding), padding

    def decode(self, target, memory, source_padding):
        causal = torch.ones(target.size(1), target.size(1), device=target.device,
                            dtype=torch.bool).triu(1)
        hidden = self.transformer.decoder(
            self.embed(target), memory, tgt_mask=causal,
            tgt_key_padding_mask=target.eq(self.pad_id),
            memory_key_padding_mask=source_padding,
        )
        return self.output(hidden)

    def forward(self, source, decoder_input):
        memory, padding = self.encode(source)
        return self.decode(decoder_input, memory, padding)

    @torch.no_grad()
    def generate_greedy(self, source, bos_id, eos_id, max_new_tokens, forbidden_ids=()):
        memory, padding = self.encode(source)
        sequence = torch.full((len(source), 1), bos_id, device=source.device, dtype=torch.long)
        done = torch.zeros(len(source), device=source.device, dtype=torch.bool)
        forbidden = sorted(set((self.pad_id, bos_id, *forbidden_ids)))
        for _ in range(min(max_new_tokens, self.settings["max_length"] - 1)):
            logits = self.decode(sequence, memory, padding)[:, -1]
            logits[:, forbidden] = -float("inf")
            token = torch.where(done, eos_id, logits.argmax(-1))
            sequence = torch.cat((sequence, token[:, None]), dim=1)
            done |= token.eq(eos_id)
            if bool(done.all()):
                break
        return sequence[:, 1:]

    @torch.no_grad()
    def generate_beam(self, source, bos_id, eos_id, max_new_tokens, beam_size=5,
                      length_penalty=1.0, forbidden_ids=()):
        """Batched beam search with finished-beam preservation and GNMT length penalty."""
        if beam_size < 1 or length_penalty <= 0:
            raise ValueError("Invalid beam settings.")
        if beam_size == 1:
            return self.generate_greedy(source, bos_id, eos_id, max_new_tokens, forbidden_ids)
        memory, padding = self.encode(source)
        batch, vocab = len(source), self.output.out_features
        sequence = torch.full((batch, 1, 1), bos_id, device=source.device, dtype=torch.long)
        scores = torch.zeros((batch, 1), device=source.device)
        lengths = torch.zeros((batch, 1), device=source.device, dtype=torch.long)
        done = torch.zeros((batch, 1), device=source.device, dtype=torch.bool)
        forbidden = sorted(set((self.pad_id, bos_id, *forbidden_ids)))
        for _ in range(min(max_new_tokens, self.settings["max_length"] - 1)):
            beams = sequence.size(1)
            expanded_memory = memory[:, None].expand(-1, beams, -1, -1).reshape(
                batch * beams, memory.size(1), memory.size(2))
            expanded_padding = padding[:, None].expand(-1, beams, -1).reshape(
                batch * beams, padding.size(1))
            logits = self.decode(sequence.reshape(batch * beams, -1),
                                 expanded_memory, expanded_padding)[:, -1]
            logits[:, forbidden] = -float("inf")
            log_probs = torch.log_softmax(logits, -1).reshape(batch, beams, vocab)
            if bool(done.any()):
                log_probs = log_probs.masked_fill(done[:, :, None], -float("inf"))
                eos_scores = log_probs[:, :, eos_id]
                log_probs[:, :, eos_id] = torch.where(done, torch.zeros_like(eos_scores), eos_scores)
            candidates = scores[:, :, None] + log_probs
            candidate_lengths = lengths[:, :, None] + (~done)[:, :, None]
            penalty = ((5.0 + candidate_lengths.float()) / 6.0).pow(length_penalty)
            rank = candidates / penalty
            _, choice = rank.reshape(batch, -1).topk(beam_size, dim=-1)
            parent, token = choice.div(vocab, rounding_mode="floor"), choice.remainder(vocab)
            gather_sequence = parent[:, :, None].expand(-1, -1, sequence.size(-1))
            sequence = torch.cat((sequence.gather(1, gather_sequence), token[:, :, None]), dim=-1)
            scores = candidates.reshape(batch, -1).gather(1, choice)
            lengths = candidate_lengths.expand(-1, -1, vocab).reshape(batch, -1).gather(1, choice)
            done = done.gather(1, parent) | token.eq(eos_id)
            if bool(done.all()):
                break
        final_penalty = ((5.0 + lengths.float()) / 6.0).pow(length_penalty)
        best = (scores / final_penalty).argmax(-1)
        return sequence[torch.arange(batch, device=source.device), best, 1:]
