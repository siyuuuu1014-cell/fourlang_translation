from __future__ import annotations

import math

import torch
from torch import nn


class TranslationTransformer(nn.Module):
    """Random-initialized, pre-norm encoder-decoder Transformer; no pretrained weights."""

    def __init__(self, vocab_size: int, config: dict, pad_id: int = 0):
        super().__init__()
        self.settings = dict(config)
        self.pad_id = pad_id
        width = config["d_model"]
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.position = nn.Embedding(config["max_length"], width)
        self.dropout = nn.Dropout(config["dropout"])
        self.transformer = nn.Transformer(
            d_model=width,
            nhead=config["nhead"],
            num_encoder_layers=config["encoder_layers"],
            num_decoder_layers=config["decoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        # nn.Transformer clones layers: explicitly initialize each layer independently.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=width**-0.5)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()

    def embed(self, ids):
        if ids.size(1) > self.settings["max_length"]:
            raise ValueError("Token sequence exceeds positional embedding capacity.")
        positions = torch.arange(ids.size(1), device=ids.device)
        return self.dropout(
            self.embedding(ids) * math.sqrt(self.settings["d_model"])
            + self.position(positions)[None]
        )

    def encode(self, source):
        padding = source.eq(self.pad_id)
        return self.transformer.encoder(
            self.embed(source), src_key_padding_mask=padding
        ), padding

    def decode(self, target, memory, source_padding):
        causal = torch.ones(
            target.size(1), target.size(1), device=target.device, dtype=torch.bool
        ).triu(1)
        hidden = self.transformer.decoder(
            self.embed(target),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=target.eq(self.pad_id),
            memory_key_padding_mask=source_padding,
        )
        return self.output(hidden)

    def forward(self, source, decoder_input):
        memory, padding = self.encode(source)
        return self.decode(decoder_input, memory, padding)

    @torch.no_grad()
    def generate(
        self, source, bos_id: int, eos_id: int, max_new_tokens: int, forbidden_ids=()
    ):
        """Batched greedy decoding, encoder computed once, explicit EOS termination."""
        memory, padding = self.encode(source)
        generated = torch.full(
            (len(source), 1), bos_id, device=source.device, dtype=torch.long
        )
        done = torch.zeros(len(source), device=source.device, dtype=torch.bool)
        for _ in range(min(max_new_tokens, self.settings["max_length"] - 1)):
            logits = self.decode(generated, memory, padding)[:, -1]
            logits[:, list(set((self.pad_id, bos_id, *forbidden_ids)))] = -float("inf")
            token = logits.argmax(-1)
            token = torch.where(done, eos_id, token)
            generated = torch.cat((generated, token[:, None]), dim=1)
            done |= token.eq(eos_id)
            if bool(done.all()):
                break
        return generated[:, 1:]
