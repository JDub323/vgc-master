"""PolicyValueNet: small from-scratch transformer encoder over the fixed token
layout. Outputs a policy over joint actions, a value in [-1, 1], and auxiliary
predictions of the opponent's hidden sets (trained on oracle team-sheet labels
to shape representations and sanity-check the particle filter — not used by
search).

The joint policy head is a 39x39 masked softmax, so each slot's action is
predicted in the context of its partner's action. Historical per-slot
checkpoints are intentionally unsupported; the frozen baseline uses this
joint-head architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from actions import N_JOINT_ACTIONS, static_joint_mask
from config import CFG, config_from_snapshot, config_snapshot


MODEL_CFG_FIELDS = ("d_model", "n_layers", "n_heads", "d_ff", "dropout")


class PolicyValueNet(nn.Module):
    """Transformer joint-policy/value/hidden-set auxiliary network."""

    def __init__(self, vocab_size, n_tokens, opp_positions,
                 n_moves, n_items, n_abilities, cfg=CFG, policy_head="joint",
                 model_cfg=None):
        super().__init__()
        if policy_head != "joint":
            raise ValueError("only joint-policy checkpoints are supported")
        model_cfg = model_cfg or {k: getattr(cfg, k) for k in MODEL_CFG_FIELDS}
        self.cfg_snapshot = config_snapshot(cfg)
        self.cfg_snapshot.update({k: model_cfg[k] for k in MODEL_CFG_FIELDS})
        self.hp = {"vocab_size": vocab_size, "n_tokens": n_tokens,
                   "opp_positions": list(opp_positions), "n_moves": n_moves,
                   "n_items": n_items, "n_abilities": n_abilities,
                   "policy_head": policy_head,
                   "model_cfg": dict(model_cfg)}
        self.policy_head = policy_head
        d = int(model_cfg["d_model"])
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        layer = nn.TransformerEncoderLayer(
            d, int(model_cfg["n_heads"]), int(model_cfg["d_ff"]),
            float(model_cfg["dropout"]),
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, int(model_cfg["n_layers"]))
        self.norm = nn.LayerNorm(d)
        self.joint_head = nn.Linear(d, N_JOINT_ACTIONS)
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
        """Return joint-policy logits, scalar value, and set predictions."""
        h = self.encoder(self.emb(tokens) + self.pos)
        cls = self.norm(h[:, 0])
        pol = self.joint_head(cls)
        value = torch.tanh(self.value_head(cls)).squeeze(-1)
        opp_h = h[:, self.opp_pos]                       # [B, 6, d]
        aux = (self.item_head(opp_h), self.ability_head(opp_h),
               self.moves_head(opp_h))
        return pol, value, aux

    def joint_dist(self, pol):
        """Normalize logits over statically legal joint actions."""
        return F.softmax(pol.masked_fill(~self.joint_mask, float("-inf")), -1)

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
        """Write hyperparameters, eager-keyed state, and config snapshot."""
        torch.save({"hp": self.hp, "state": clean_state_dict(self),
                    "cfg": self.cfg_snapshot}, path)

    @classmethod
    def load(cls, path, cfg=CFG, device="cpu"):
        """Return a checkpoint-restored model on ``device``."""
        ck = torch.load(path, map_location=device, weights_only=False)
        load_cfg = config_from_snapshot(ck.get("cfg"), base=cfg)
        hp = dict(ck["hp"])
        hp.setdefault("model_cfg", _model_cfg_from_checkpoint(hp, ck["state"], load_cfg))
        m = cls(**hp, cfg=load_cfg).to(device)
        m.load_state_dict(strip_compile_prefix(ck["state"]))
        return m

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


def _model_cfg_from_checkpoint(hp, state, cfg):
    """Best-effort architecture recovery for checkpoints saved before cfg.

    n_heads and dropout are not encoded in old state dicts, so those remain
    supplied by cfg unless the checkpoint stores model_cfg.
    """
    state = strip_compile_prefix(state)
    d_model = state["emb.weight"].shape[1] if "emb.weight" in state else cfg.d_model
    layer_ids = [
        int(k.split(".")[2]) for k in state
        if k.startswith("encoder.layers.") and k.split(".")[2].isdigit()
    ]
    d_ff = state["encoder.layers.0.linear1.weight"].shape[0] \
        if "encoder.layers.0.linear1.weight" in state else cfg.d_ff
    return {
        "d_model": int(d_model),
        "n_layers": max(layer_ids) + 1 if layer_ids else cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_ff": int(d_ff),
        "dropout": cfg.dropout,
    }
