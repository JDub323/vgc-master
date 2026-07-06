"""PolicyValueNet: small from-scratch transformer encoder over the fixed token
layout. Outputs per-slot action logits (the doubles action space is factorized
per slot; search recombines), a value in [-1, 1], and auxiliary predictions of
the opponent's hidden sets (trained on oracle team-sheet labels to shape
representations and sanity-check the particle filter — not used by search).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from actions import N_SLOT_ACTIONS
from config import CFG


class PolicyValueNet(nn.Module):
    def __init__(self, vocab_size, n_tokens, opp_positions,
                 n_moves, n_items, n_abilities, cfg=CFG):
        super().__init__()
        self.hp = {"vocab_size": vocab_size, "n_tokens": n_tokens,
                   "opp_positions": list(opp_positions), "n_moves": n_moves,
                   "n_items": n_items, "n_abilities": n_abilities}
        d = cfg.d_model
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.d_ff, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(d)
        self.slot_heads = nn.Linear(d, 2 * N_SLOT_ACTIONS)
        self.value_head = nn.Linear(d, 1)
        self.register_buffer("opp_pos", torch.tensor(list(opp_positions)))
        self.item_head = nn.Linear(d, n_items + 1)
        self.ability_head = nn.Linear(d, n_abilities + 1)
        self.moves_head = nn.Linear(d, n_moves + 1)

    def forward(self, tokens):
        h = self.encoder(self.emb(tokens) + self.pos)
        cls = self.norm(h[:, 0])
        slots = self.slot_heads(cls).view(-1, 2, N_SLOT_ACTIONS)
        value = torch.tanh(self.value_head(cls)).squeeze(-1)
        opp_h = h[:, self.opp_pos]                       # [B, 6, d]
        aux = (self.item_head(opp_h), self.ability_head(opp_h),
               self.moves_head(opp_h))
        return slots, value, aux

    @torch.no_grad()
    def predict_batch(self, tokens):
        """tokens: [B, n_tokens] int tensor/array. Returns
        (per_slot_action_dists [B,2,A], values [B], set_predictions) as numpy —
        batched for use inside search."""
        self.eval()
        dev = next(self.parameters()).device
        t = torch.as_tensor(tokens, dtype=torch.long, device=dev)
        slots, value, (items, abils, moves) = self(t)
        return (F.softmax(slots, dim=-1).cpu().numpy(),
                value.cpu().numpy(),
                {"items": F.softmax(items, -1).cpu().numpy(),
                 "abilities": F.softmax(abils, -1).cpu().numpy(),
                 "moves": torch.sigmoid(moves).cpu().numpy()})

    def save(self, path):
        torch.save({"hp": self.hp, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, cfg=CFG, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(**ck["hp"], cfg=cfg).to(device)
        m.load_state_dict(ck["state"])
        return m
