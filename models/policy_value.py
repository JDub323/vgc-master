"""PolicyValueNet: small from-scratch transformer encoder over the fixed token
layout. Outputs a policy over joint actions, a value in [-1, 1], and auxiliary
predictions of the opponent's hidden sets (trained on oracle team-sheet labels
to shape representations and sanity-check the particle filter — not used by
search).

Two policy-head architectures, selected by hp["policy_head"]:

  "slot"  (v1) — two independent 39-way softmaxes, one per slot, recombined
          into a joint distribution by outer product. Checkpoints trained
          before the joint head load and play exactly as before.
  "joint" (v2) — one 39x39 masked softmax over joint actions. A slot's action
          is predicted in the context of its partner's action, which the
          factorized head cannot express (e.g. an attack that only makes
          sense under partner's Rage Powder).

Everything downstream consumes one contract: predict_batch returns a
normalized joint distribution [B, N_JOINT_ACTIONS] for either head, so search,
evaluation and benchmarking are architecture-blind.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from actions import N_JOINT_ACTIONS, N_SLOT_ACTIONS, static_joint_mask
from config import CFG


class PolicyValueNet(nn.Module):
    def __init__(self, vocab_size, n_tokens, opp_positions,
                 n_moves, n_items, n_abilities, cfg=CFG, policy_head="slot"):
        super().__init__()
        # "slot" default keeps **hp construction of old checkpoints working;
        # new models are built with policy_head="joint" explicitly.
        assert policy_head in ("slot", "joint"), policy_head
        self.hp = {"vocab_size": vocab_size, "n_tokens": n_tokens,
                   "opp_positions": list(opp_positions), "n_moves": n_moves,
                   "n_items": n_items, "n_abilities": n_abilities,
                   "policy_head": policy_head}
        self.policy_head = policy_head
        d = cfg.d_model
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.d_ff, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(d)
        if policy_head == "joint":
            self.joint_head = nn.Linear(d, N_JOINT_ACTIONS)
        else:
            self.slot_heads = nn.Linear(d, 2 * N_SLOT_ACTIONS)
        self.value_head = nn.Linear(d, 1)
        self.register_buffer("opp_pos", torch.tensor(list(opp_positions)))
        # persistent=False: deterministic from actions.py, kept out of
        # state_dict so v1 checkpoints (which predate it) load strictly
        self.register_buffer("joint_mask",
                             torch.from_numpy(static_joint_mask().reshape(-1)),
                             persistent=False)
        self.item_head = nn.Linear(d, n_items + 1)
        self.ability_head = nn.Linear(d, n_abilities + 1)
        self.moves_head = nn.Linear(d, n_moves + 1)

    def forward(self, tokens):
        """Returns (policy logits, value, aux). Policy logits are
        [B, N_JOINT_ACTIONS] for the joint head, [B, 2, N_SLOT_ACTIONS] for
        the slot head — training branches on the shape, inference goes
        through joint_dist() and never sees the difference."""
        h = self.encoder(self.emb(tokens) + self.pos)
        cls = self.norm(h[:, 0])
        if self.policy_head == "joint":
            pol = self.joint_head(cls)
        else:
            pol = self.slot_heads(cls).view(-1, 2, N_SLOT_ACTIONS)
        value = torch.tanh(self.value_head(cls)).squeeze(-1)
        opp_h = h[:, self.opp_pos]                       # [B, 6, d]
        aux = (self.item_head(opp_h), self.ability_head(opp_h),
               self.moves_head(opp_h))
        return pol, value, aux

    def joint_dist(self, pol):
        """Policy logits (either head) -> normalized joint distribution
        [B, N_JOINT_ACTIONS] over statically-legal joint actions."""
        if self.policy_head == "joint":
            return F.softmax(pol.masked_fill(~self.joint_mask, float("-inf")), -1)
        pa, pb = F.softmax(pol[:, 0], -1), F.softmax(pol[:, 1], -1)
        j = (pa[:, :, None] * pb[:, None, :]).flatten(1) * self.joint_mask
        return j / j.sum(-1, keepdim=True)

    @torch.no_grad()
    def predict_batch(self, tokens):
        """tokens: [B, n_tokens] int tensor/array. Returns
        (joint action dists [B, N_JOINT_ACTIONS], values [B],
        set_predictions) as numpy — batched for use inside search."""
        self.eval()
        dev = next(self.parameters()).device
        t = torch.as_tensor(tokens, dtype=torch.long, device=dev)
        pol, value, (items, abils, moves) = self(t)
        return (self.joint_dist(pol).cpu().numpy(),
                value.cpu().numpy(),
                {"items": F.softmax(items, -1).cpu().numpy(),
                 "abilities": F.softmax(abils, -1).cpu().numpy(),
                 "moves": torch.sigmoid(moves).cpu().numpy()})

    def save(self, path):
        torch.save({"hp": self.hp, "state": clean_state_dict(self)}, path)

    @classmethod
    def load(cls, path, cfg=CFG, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(**ck["hp"], cfg=cfg).to(device)
        m.load_state_dict(strip_compile_prefix(ck["state"]))
        return m

    @classmethod
    def from_slot(cls, slot_model, cfg=CFG):
        """Convert a v1 per-slot model into a joint-head model that computes
        the SAME joint distribution at conversion time: joint logit(a, b) =
        slot_a logit(a) + slot_b logit(b), which softmaxes to the outer
        product. A warm start for fine-tuning (self-play or BC) without
        retraining the trunk from scratch."""
        assert slot_model.policy_head == "slot"
        hp = dict(slot_model.hp)
        hp["policy_head"] = "joint"
        m = cls(**hp, cfg=cfg)
        trunk = {k: v for k, v in slot_model.state_dict().items()
                 if not k.startswith("slot_heads")}
        m.load_state_dict(trunk, strict=False)
        W, b = slot_model.slot_heads.weight.detach(), slot_model.slot_heads.bias.detach()
        A = N_SLOT_ACTIONS
        wa, wb = W[:A], W[A:]                            # [A, d] each
        m.joint_head.weight.data.copy_(
            (wa.unsqueeze(1) + wb.unsqueeze(0)).reshape(A * A, -1))
        m.joint_head.bias.data.copy_(
            (b[:A].unsqueeze(1) + b[A:].unsqueeze(0)).reshape(-1))
        return m.to(next(slot_model.parameters()).device)


def clean_state_dict(model):
    """State dict with eager keys regardless of torch.compile. Compiling wraps
    the module in OptimizedModule and prefixes every key with '_orig_mod.',
    which poisons checkpoints: they can then only be loaded back into a
    compiled wrapper. Saving through the underlying module keeps every
    checkpoint loadable everywhere (search, evaluate, benchmark, resume)."""
    return getattr(model, "_orig_mod", model).state_dict()


def strip_compile_prefix(state):
    """Accept checkpoints saved from a compiled model before this fix."""
    p = "_orig_mod."
    return {k[len(p):] if k.startswith(p) else k: v for k, v in state.items()}
