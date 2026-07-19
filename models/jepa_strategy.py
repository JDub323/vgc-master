"""JEPAStrategyModel (v3 stage 1): recursive latent dynamics for VGC.

The core object is a latent transition ``T(Z, a, b) -> Z'`` over the full
16-entity latent set, conditioned on BOTH sides' joint actions. Unlike v2's
terminal consequence vector, ``T``'s output lives in the same space as its
input, so it can be applied repeatedly — latent lookahead without touching the
simulator. Training is multi-step JEPA: unroll ``T`` along real trajectories
and match each step against an EMA target encoder's embedding of the realized
position (``train_strategy.py``).

Heads: own-prior and opponent-policy per-slot logits (candidate generators for
the matrix search), and a scalar value read off any latent set — encoded or
predicted — so a payoff matrix is one batched ``T`` application away
(``agents/jepa_world_model/v3.py``). Distributional value and the strategy
bottleneck arrive in later stages (see ``JEPA_V3_DESIGN.md``).
"""

import copy

import torch
import torch.nn as nn

from jepa.features import MON_TOKENS, N_MON
from models.jepa_wm import (ALLY0, CLS_TOK, DamageBias, Encoder, N_ROLES,
                            RoleTypedLayer)
from models.jepa_wm import N_SLOT_ACTIONS


class Dynamics(nn.Module):
    """Latent transition: (entity latents, both joint actions) -> next latents.

    Same input/output space, so it composes: ``T(T(Z, a0, b0), a1, b1)`` is a
    two-turn latent rollout. The damage-edge attention bias is optional —
    supplied at the first step (current-state edges are known) and omitted on
    deeper recursive steps where no edge matrix exists."""

    def __init__(self, emb, jcfg):
        """Build action tables (sharing the encoder's move table) + layers."""
        super().__init__()
        de, d = jcfg.d_embed, jcfg.d_model
        self.emb = [emb]                     # shared move table, not a submodule
        self.kind = nn.Embedding(3, de)
        self.target = nn.Embedding(4, de)
        self.switch = nn.Embedding(N_MON + 1, de)
        self.act_proj = nn.Linear(4 * de + 3, d)
        self.dmg_bias = DamageBias(jcfg)
        self.layers = nn.ModuleList(RoleTypedLayer(jcfg)
                                    for _ in range(jcfg.n_cpred_layers))
        self.role = nn.Embedding(N_ROLES, d)
        from jepa.features import ROLES
        self.register_buffer("roles", torch.tensor(ROLES), persistent=False)
        self.out_norm = nn.LayerNorm(d)

    def action_embed(self, act):
        """``[B,12,7]`` both-sides action arrays -> ``[B,12,d]`` embeddings."""
        move_tab = self.emb[0].move
        cat = torch.cat([self.kind(act[..., 0]), move_tab(act[..., 1]),
                         self.target(act[..., 2]), self.switch(act[..., 4])], -1)
        scal = torch.stack([act[..., 3].float(), act[..., 5].float() / 250.0,
                            act[..., 6].float() / 12.0], -1)
        return self.act_proj(torch.cat([cat, scal], -1))

    def forward(self, z, act, dmg=None):
        """Advance latents one turn under joint actions ``act`` (both sides)."""
        x = z + self.role(self.roles).unsqueeze(0)
        a = self.action_embed(act)
        x = x.clone()
        x[:, ALLY0:ALLY0 + MON_TOKENS] = x[:, ALLY0:ALLY0 + MON_TOKENS] + a
        bias = self.dmg_bias(dmg) if dmg is not None else None
        for layer in self.layers:
            x = layer(x, bias)
        return self.out_norm(x)


class JEPAStrategyModel(nn.Module):
    """Encoder + recursive dynamics + policy/value heads + EMA target."""

    def __init__(self, sizes, jcfg, vocab_state=None):
        """Assemble online/target encoders, dynamics ``T``, and readout heads."""
        super().__init__()
        self.jcfg = jcfg
        self.sizes = dict(sizes)
        self.vocab_state = vocab_state
        d = jcfg.d_model
        self.encoder = Encoder(sizes, jcfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.dynamics = Dynamics(self.encoder.embedder, jcfg)
        self.my_prior = nn.Linear(d, 2 * N_SLOT_ACTIONS)
        self.opp_policy = nn.Linear(d, 2 * N_SLOT_ACTIONS)
        self.value_head = nn.Linear(d, 1)

    # -- encode / step -------------------------------------------------------
    def encode(self, pos):
        """Online-encode a position tensor dict to entity latents ``[B,16,d]``."""
        return self.encoder(pos["gcat"], pos["gscal"], pos["mcat"],
                            pos["mscal"], pos["dmg"])

    @torch.no_grad()
    def target_encode(self, pos):
        """EMA-encode a position to full target latents (stop-grad)."""
        return self.target_encoder(pos["gcat"], pos["gscal"], pos["mcat"],
                                   pos["mscal"], pos["dmg"])

    def step(self, z, act, dmg=None):
        """Apply the dynamics once: ``T(Z, both-actions) -> Z'``."""
        return self.dynamics(z, act, dmg)

    # -- heads ---------------------------------------------------------------
    def value(self, z):
        """Scalar value in [-1,1] read off a latent set's CLS token."""
        return torch.tanh(self.value_head(z[:, CLS_TOK])).squeeze(-1)

    def policies(self, z):
        """Return ``(my_prior, opp_policy)`` slot logits ``[B, 2, 39]``."""
        cls, intent = z[:, CLS_TOK], z[:, 13:15].mean(1)
        my = self.my_prior(cls).view(-1, 2, N_SLOT_ACTIONS)
        opp = self.opp_policy(intent).view(-1, 2, N_SLOT_ACTIONS)
        return my, opp

    @torch.no_grad()
    def update_ema(self):
        """EMA-update the target encoder toward the online encoder."""
        m = self.jcfg.ema_decay
        for tp, p in zip(self.target_encoder.parameters(),
                         self.encoder.parameters()):
            tp.mul_(m).add_(p, alpha=1 - m)
        for tb, b in zip(self.target_encoder.buffers(),
                         self.encoder.buffers()):
            tb.copy_(b)

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        """Write sizes, config, vocab id-space, and weights to ``path``."""
        import dataclasses
        torch.save({"sizes": self.sizes,
                    "jcfg": dataclasses.asdict(self.jcfg),
                    "vocab_state": self.vocab_state,
                    "kind": "strategy",
                    "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a checkpoint, returning ``(model, vocab_state_or_None)``."""
        from jepa.config import JEPAConfig
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck.get("kind") != "strategy":
            raise ValueError("not a JEPA-Strategy checkpoint")
        jcfg = JEPAConfig(**{k: v for k, v in ck["jcfg"].items()
                             if k in JEPAConfig.__dataclass_fields__})
        m = cls(ck["sizes"], jcfg, ck.get("vocab_state")).to(device)
        m.load_state_dict(ck["state"])
        return m, ck.get("vocab_state")
