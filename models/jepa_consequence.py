"""JEPAConsequenceModel: pure latent move-consequence prediction for VGC.

The corrected JEPA formulation (see ``JEPA_DESIGN.md`` "Consequence variant").
For the current position and each candidate OWN joint move, the model predicts
a single latent **consequence vector** that summarizes the distribution over
what happens after the opponent responds and chance resolves. It is never
decoded to an explicit next state; it only has to be sufficient to compare
moves.

Training is JEPA-style: the consequence vector of the move actually taken is
matched (smooth-L1) to an EMA target-encoder's embedding of the realized future
position ``horizon`` plies later. Because the same (position, move) maps to many
different futures across the data (and across simulated what-ifs), the predictor
is forced to output the *expected* future embedding — implicitly learning the
engine, opponent behavior, and stochastic effects. An optional luck latent
``xi`` lets the predictor represent the spread, not just the mean.

Heads read only the consequence vector: a policy head ranks candidate moves
(behavior cloning on strong-human choices) and a value head predicts the game
outcome. Neither reconstructs board state.
"""

import copy

import torch
import torch.nn as nn

from jepa.features import MON_TOKENS, N_MON
from models.jepa_wm import (ALLY0, CLS_TOK, DamageBias, Encoder, RoleTypedLayer)


class ConsequencePredictor(nn.Module):
    """(context latents, own move, luck) -> one latent consequence vector."""

    def __init__(self, emb, jcfg):
        """Build own-move action tables + predictor layers over the encoder emb."""
        super().__init__()
        de, d = jcfg.d_embed, jcfg.d_model
        self.emb = [emb]                      # shared move table (not a submodule)
        self.kind = nn.Embedding(3, de)
        self.target = nn.Embedding(4, de)
        self.switch = nn.Embedding(N_MON + 1, de)
        self.act_proj = nn.Linear(4 * de + 3, d)
        self.noise_dim = jcfg.noise_dim
        if self.noise_dim:
            self.noise_proj = nn.Linear(self.noise_dim, d)
        self.dmg_bias = DamageBias(jcfg)
        self.layers = nn.ModuleList(RoleTypedLayer(jcfg)
                                    for _ in range(jcfg.n_cpred_layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def action_embed(self, act):
        """Own action arrays ``[B,12,7]`` -> per-mon action embeddings ``[B,12,d]``."""
        move_tab = self.emb[0].move
        cat = torch.cat([self.kind(act[..., 0]), move_tab(act[..., 1]),
                         self.target(act[..., 2]), self.switch(act[..., 4])], -1)
        scal = torch.stack([act[..., 3].float(), act[..., 5].float() / 250.0,
                            act[..., 6].float() / 12.0], -1)
        return self.act_proj(torch.cat([cat, scal], -1))

    def forward(self, z, my_act, dmg_edge, xi=None):
        """Predict the consequence vector ``[B, d]`` for own move ``my_act``."""
        x = z.clone()
        a = self.action_embed(my_act)                      # [B,12,d]
        x[:, ALLY0:ALLY0 + MON_TOKENS] = x[:, ALLY0:ALLY0 + MON_TOKENS] + a
        if self.noise_dim and xi is not None:
            x[:, CLS_TOK] = x[:, CLS_TOK] + self.noise_proj(xi)
        bias = self.dmg_bias(dmg_edge)
        for layer in self.layers:
            x = layer(x, bias)
        return self.out(x[:, CLS_TOK])                     # [B, d]


class JEPAConsequenceModel(nn.Module):
    """Encoder + consequence predictor + policy/value heads + EMA target."""

    def __init__(self, sizes, jcfg, vocab_state=None):
        """Assemble the online/target encoders, predictor, and readout heads."""
        super().__init__()
        self.jcfg = jcfg
        self.sizes = dict(sizes)
        self.vocab_state = vocab_state
        d = jcfg.d_model
        self.encoder = Encoder(sizes, jcfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = ConsequencePredictor(self.encoder.embedder, jcfg)
        self.policy_head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                         nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                        nn.Linear(d, 1))

    # -- encode/predict ------------------------------------------------------
    def encode(self, pos):
        """Online-encode a position tensor dict to entity latents ``[B,16,d]``."""
        return self.encoder(pos["gcat"], pos["gscal"], pos["mcat"],
                            pos["mscal"], pos["dmg"])

    @torch.no_grad()
    def target_context(self, pos):
        """EMA-encode a (future) position and pool the CLS token (stop-grad)."""
        z = self.target_encoder(pos["gcat"], pos["gscal"], pos["mcat"],
                                pos["mscal"], pos["dmg"])
        return z[:, CLS_TOK]

    def consequence(self, z, my_act, dmg, xi=None):
        """Predict the latent consequence vector of one own move."""
        return self.predictor(z, my_act, dmg, xi)

    def score(self, c):
        """Policy-head desirability logit for a consequence vector ``[B]``."""
        return self.policy_head(c).squeeze(-1)

    def value(self, c):
        """Win-probability value in [-1,1] read off a consequence vector."""
        return torch.tanh(self.value_head(c)).squeeze(-1)

    def sample_noise(self, b, device):
        """Draw a luck latent ``[B, noise_dim]`` (empty when deterministic)."""
        if not self.jcfg.noise_dim:
            return None
        return torch.randn(b, self.jcfg.noise_dim, device=device)

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
                    "kind": "consequence",
                    "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a checkpoint, returning ``(model, vocab_state_or_None)``."""
        from jepa.config import JEPAConfig
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck.get("kind") != "consequence":
            raise ValueError("not a JEPA-Consequence checkpoint")
        jcfg = JEPAConfig(**{k: v for k, v in ck["jcfg"].items()
                             if k in JEPAConfig.__dataclass_fields__})
        m = cls(ck["sizes"], jcfg, ck.get("vocab_state")).to(device)
        m.load_state_dict(ck["state"])
        return m, ck.get("vocab_state")
