"""Experimental leaf-value architectures for the exp/value-head experiment.

The baseline PolicyValueNet reads its value from one ``Linear(d, 1) + tanh``
on the CLS state, trained with MSE against the +-1 final outcome. This module
provides drop-in replacements for that scalar only — the policy and aux heads
are never touched — attacking three documented weaknesses:

1. **CLS bottleneck.** The value must squeeze through the single 256-d vector
   the policy and aux heads also train against. The heads here read the full
   ``[B, n_tokens, d]`` encoder state directly (attention pooling), so the
   value path can look at per-mon blocks, belief tokens, and the damage matrix
   without competing for CLS capacity.
2. **tanh + MSE saturation.** With MSE through tanh, a confidently *wrong*
   value (v -> +-1, z opposite) receives a vanishing gradient — exactly the
   worst predictions get the least correction. The default output here is a
   win *logit* trained with cross-entropy ("value as classification":
   KataGo, MuZero, Farebrother et al. 2024), whose gradient does not saturate
   and which calibrates better — which matters because search mixes these
   values with exact +-1 terminal values.
3. **Head capacity.** A linear readout is the smallest possible head; the
   replacements are small MLPs, optionally on pooled features.

Two integration shapes, both packaged behind ``ValueAugmentedNet``:

- ``kind="head"``: a head over the **frozen** baseline trunk's states — zero
  extra forward passes at search leaves.
- ``kind="net"``: a **dedicated value network** (baseline-architecture trunk,
  initialized from the baseline, fine-tuned end-to-end on the value objective
  only) — one extra forward per leaf batch.

``ValueAugmentedNet.predict_batch`` mirrors
``models.policy_value.PolicyValueNet.predict_batch`` exactly, so the combined
model drops into ``PolicyValueLeafEvaluator`` / ``DeterminizedDUCTChooser``
unchanged: joint-action distributions and aux predictions are bit-identical to
the frozen baseline; only the value scalar changes. A combined checkpoint
(``save_combined`` / ``load_value_agent``) embeds the full baseline checkpoint
so one file is a complete agent for ``agent_server.py --agent search-vh``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, config_from_snapshot
from models.policy_value import (PolicyValueNet, _model_cfg_from_checkpoint,
                                 strip_compile_prefix)

COMBINED_SCHEMA = "value-combined-v1"
N_AUX_MARGINS = 2          # (faint-differential, hp-sum-differential) targets


def _mlp(d_in, hidden, depth, d_out, dropout):
    """Return an MLP ``d_in -> hidden^depth -> d_out`` with a zero-init out.

    The zero-initialized final layer starts every candidate at v=0 ("even
    position"), which keeps early training stable and means an untrained head
    is harmless rather than random."""
    layers, prev = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
        prev = hidden
    out = nn.Linear(prev, d_out)
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class CLSMLPHead(nn.Module):
    """LayerNorm + MLP value head over the CLS state only.

    The cheapest capacity upgrade: same input as the baseline's linear head,
    strictly more expressive. Output ``[B, 1 + n_aux]``: win logit first, then
    auxiliary margin predictions."""

    def __init__(self, d, hidden=512, depth=2, dropout=0.1, n_aux=N_AUX_MARGINS):
        """Build the head; ``d`` is the trunk width."""
        super().__init__()
        self.hp = {"arch": "clsmlp", "d": d, "hidden": hidden, "depth": depth,
                   "dropout": dropout, "n_aux": n_aux}
        self.norm = nn.LayerNorm(d)
        self.mlp = _mlp(d, hidden, depth, 1 + n_aux, dropout)

    def forward(self, h):
        """``[B, T, d]`` trunk states -> ``[B, 1 + n_aux]`` outputs."""
        return self.mlp(self.norm(h[:, 0]))


class AttnPoolHead(nn.Module):
    """Learned-query attention pooling over all token states + MLP.

    ``n_queries`` learned queries cross-attend over the full 561-token encoder
    state (KataGo-style global pooling adapted to a set-of-tokens trunk); the
    query outputs are concatenated with a mean-pool and fed to an MLP. This
    gives the value path direct access to every mon block, belief token, and
    damage-matrix token instead of only what the trunk compressed into CLS."""

    def __init__(self, d, n_queries=4, n_heads=8, hidden=512, dropout=0.1,
                 n_aux=N_AUX_MARGINS):
        """Build queries, one MultiheadAttention, and the readout MLP."""
        super().__init__()
        self.hp = {"arch": "attnpool", "d": d, "n_queries": n_queries,
                   "n_heads": n_heads, "hidden": hidden, "dropout": dropout,
                   "n_aux": n_aux}
        self.norm = nn.LayerNorm(d)
        self.query = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                          batch_first=True)
        self.mlp = _mlp((n_queries + 1) * d, hidden, 2, 1 + n_aux, dropout)

    def forward(self, h):
        """``[B, T, d]`` trunk states -> ``[B, 1 + n_aux]`` outputs."""
        x = self.norm(h)
        q = self.query.expand(len(x), -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)      # [B, Q, d]
        feats = torch.cat([pooled.flatten(1), x.mean(dim=1)], dim=-1)
        return self.mlp(feats)


HEAD_BUILDERS = {"clsmlp": CLSMLPHead, "attnpool": AttnPoolHead}


def build_head(hp):
    """Rebuild a head module from its recorded ``hp`` dict."""
    hp = dict(hp)
    return HEAD_BUILDERS[hp.pop("arch")](**hp)


class ValueNet(nn.Module):
    """Dedicated value network: baseline-architecture trunk + pooled head.

    The trunk (embedding, positional table, encoder) is dimensioned from the
    baseline's ``model_cfg`` and normally initialized from the baseline's
    weights (``from_base``), then fine-tuned end-to-end on the value objective
    alone — the "seq2seq-shaped dedicated evaluator" candidate. At play time
    it costs one extra forward per leaf batch."""

    def __init__(self, vocab_size, n_tokens, model_cfg, head="attnpool",
                 n_aux=N_AUX_MARGINS):
        """Build trunk modules and the value head from plain dimensions."""
        super().__init__()
        self.hp = {"vocab_size": vocab_size, "n_tokens": n_tokens,
                   "model_cfg": dict(model_cfg), "head": head, "n_aux": n_aux}
        d = int(model_cfg["d_model"])
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        layer = nn.TransformerEncoderLayer(
            d, int(model_cfg["n_heads"]), int(model_cfg["d_ff"]),
            float(model_cfg["dropout"]),
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, int(model_cfg["n_layers"]))
        self.head = HEAD_BUILDERS[head](d, n_aux=n_aux)

    def forward(self, tokens):
        """``[B, T]`` token ids -> ``[B, 1 + n_aux]`` outputs."""
        return self.head(self.encoder(self.emb(tokens) + self.pos))

    @classmethod
    def from_base(cls, base, head="attnpool", n_aux=N_AUX_MARGINS):
        """Return a ValueNet whose trunk is a copy of ``base``'s trunk."""
        m = cls(base.hp["vocab_size"], base.hp["n_tokens"],
                base.hp["model_cfg"], head=head, n_aux=n_aux)
        m.emb.load_state_dict(base.emb.state_dict())
        m.pos.data.copy_(base.pos.data)
        m.encoder.load_state_dict(base.encoder.state_dict())
        return m


def value_from_logit(logit, output="bce", temperature=1.0):
    """Map a win logit to a value in ``[-1, 1]``.

    ``output="bce"``: v = 2*sigmoid(logit/T) - 1 (v is 2*P(win) - 1).
    ``output="mse"``: v = tanh(logit/T) (the ablation candidates' semantics).
    ``temperature`` is post-hoc calibration fitted on the validation split; it
    is monotone, so ordering metrics are unchanged while the values search
    mixes with exact +-1 terminal leaves become honest probabilities."""
    t = logit / float(temperature)
    return 2.0 * torch.sigmoid(t) - 1.0 if output == "bce" else torch.tanh(t)


class ValueAugmentedNet(nn.Module):
    """Frozen baseline policy/aux with an experimental value module swapped in.

    ``predict_batch`` reproduces the baseline's joint distributions and aux
    outputs bit-for-bit (same modules, same forward) and replaces only the
    value scalar, so any Elo change downstream is attributable to the value
    brick alone."""

    def __init__(self, base, value_module, kind, output="bce", temperature=1.0):
        """Freeze ``base``; ``kind`` is "head" (trunk states) or "net"."""
        super().__init__()
        assert kind in ("head", "net"), kind
        self.base, self.value_module = base, value_module
        self.kind, self.output = kind, output
        self.temperature = float(temperature)
        for p in self.base.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict_batch(self, tokens):
        """Mirror ``PolicyValueNet.predict_batch``: one shared trunk forward
        for the baseline policy/aux outputs, value from the swapped module.
        Returns ``(joint dists [B, N_JOINT_ACTIONS], values [B], aux dict)``
        as numpy."""
        self.eval()
        base = self.base
        dev = next(base.parameters()).device
        t = torch.as_tensor(tokens, dtype=torch.long, device=dev)
        h = base.encoder(base.emb(t) + base.pos)
        pol = base.joint_head(base.norm(h[:, 0]))
        opp_h = h[:, base.opp_pos]
        items, abils, moves = (base.item_head(opp_h), base.ability_head(opp_h),
                               base.moves_head(opp_h))
        out = self.value_module(h) if self.kind == "head" \
            else self.value_module(t)
        value = value_from_logit(out[:, 0], self.output, self.temperature)
        return (base.joint_dist(pol).cpu().numpy(),
                value.cpu().numpy(),
                {"items": F.softmax(items, -1).cpu().numpy(),
                 "abilities": F.softmax(abils, -1).cpu().numpy(),
                 "moves": torch.sigmoid(moves).cpu().numpy()})


def save_combined(path, base_ckpt, value_module, kind, output, temperature,
                  meta=None):
    """Write one self-contained combined checkpoint.

    ``base_ckpt`` is the *raw loaded dict* of the baseline checkpoint
    (hp/state/cfg), embedded whole so the bundle exporter's single
    ``artifacts/checkpoints/ckpt.pt`` slot carries the complete agent."""
    torch.save({"schema": COMBINED_SCHEMA,
                "base": base_ckpt,
                "value": {"kind": kind, "output": output,
                          "hp": value_module.hp,
                          "state": value_module.state_dict()},
                "temperature": float(temperature),
                "meta": dict(meta or {})}, path)


def _base_from_dict(ck, cfg, device):
    """Rebuild the frozen baseline model from an embedded checkpoint dict
    (the same restoration path as ``PolicyValueNet.load``)."""
    load_cfg = config_from_snapshot(ck.get("cfg"), base=cfg)
    hp = dict(ck["hp"])
    hp.setdefault("model_cfg",
                  _model_cfg_from_checkpoint(hp, ck["state"], load_cfg))
    m = PolicyValueNet(**hp, cfg=load_cfg).to(device)
    m.load_state_dict(strip_compile_prefix(ck["state"]))
    return m


def build_value_module(value_spec, base):
    """Instantiate the recorded value module (head or dedicated net)."""
    if value_spec["kind"] == "head":
        return build_head(value_spec["hp"])
    return ValueNet(**value_spec["hp"])


def load_value_agent(path, cfg=CFG, device="cpu"):
    """Load a combined checkpoint into a ready ``ValueAugmentedNet``."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("schema") != COMBINED_SCHEMA:
        raise ValueError(
            f"{path} is not a {COMBINED_SCHEMA} checkpoint — build one with "
            "'python value_lab.py select' (a plain policy/value checkpoint "
            "belongs to --agent search, not search-vh)")
    base = _base_from_dict(ck["base"], cfg, device)
    spec = ck["value"]
    module = build_value_module(spec, base).to(device)
    module.load_state_dict(spec["state"])
    model = ValueAugmentedNet(base, module, spec["kind"], spec["output"],
                              ck.get("temperature", 1.0)).to(device)
    model.eval()
    return model
