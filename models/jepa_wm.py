"""JEPAWorldModel: a latent one-ply world model for VGC doubles.

Pieces (see ``JEPA_DESIGN.md``):
  * Embedder      — entity features -> 16 token vectors (role-tagged).
  * Encoder       — role-typed transformer -> context latents Z.
  * Predictor     — Z + joint action (mine, opp) -> predicted next latents Ẑ'.
  * heads         — value (the payoff read off Ẑ'), grounded decoders
                    (next HP/faint/status/field/order, for grounding +
                    anti-collapse), and my/opponent policy priors off Z.
  * target Encoder — EMA copy that embeds the true next state for the JEPA
                    latent-matching target (stop-grad).

"Role-typed" attention gives allies, foes, global, intent, and CLS tokens
their own Q/K/V/O projections, and biases attention logits by an ally->foe
damage edge feature. Nothing here is imported by the frozen trunk model.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.features import (MON_TOKENS, N_ACT_FIELDS, N_GLOBAL_SCALAR,
                           N_MON, N_MON_CAT, N_MON_SCALAR, N_TOKENS, ROLES)
from jepa.vocab import STATUSES, TERRAINS, WEATHERS

N_ROLES = 5
N_SLOT_ACTIONS = 39
GLOBAL_TOK, ALLY0, FOE0, CLS_TOK = 0, 1, 7, 15


class Embedder(nn.Module):
    """Feature arrays -> ``[B, 16, d]`` role-tagged token embeddings."""

    def __init__(self, sizes, jcfg):
        """Build categorical tables and projections from vocab ``sizes``."""
        super().__init__()
        de, d = jcfg.d_embed, jcfg.d_model
        self.species = nn.Embedding(sizes["species"], de)
        self.item = nn.Embedding(sizes["item"], de)
        self.ability = nn.Embedding(sizes["ability"], de)
        self.move = nn.Embedding(sizes["move"], de)
        self.type_emb = nn.Embedding(sizes["type"], de)
        self.status = nn.Embedding(sizes["status"], de)
        self.weather = nn.Embedding(sizes["weather"], de)
        self.terrain = nn.Embedding(sizes["terrain"], de)
        self.mon_cat_proj = nn.Linear(6 * de, d)
        self.mon_scalar_proj = nn.Linear(N_MON_SCALAR, d)
        self.global_cat_proj = nn.Linear(2 * de, d)
        self.global_scalar_proj = nn.Linear(N_GLOBAL_SCALAR, d)
        self.intent = nn.Parameter(torch.zeros(2, d))
        self.cls = nn.Parameter(torch.zeros(1, d))
        self.role = nn.Embedding(N_ROLES, d)
        self.register_buffer("roles", torch.tensor(ROLES), persistent=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, gcat, gscal, mcat, mscal):
        """Map global/mon feature tensors to ``[B, 16, d]`` embeddings."""
        b = gcat.shape[0]
        # mon tokens (12)
        sp = self.species(mcat[..., 0])
        it = self.item(mcat[..., 1])
        ab = self.ability(mcat[..., 2])
        mv = self.move(mcat[..., 3:7]).mean(-2)
        ty = self.type_emb(mcat[..., 8:10]).sum(-2)
        st = self.status(mcat[..., 7])
        mon_cat = torch.cat([sp, it, ab, mv, ty, st], -1)
        mon = self.mon_cat_proj(mon_cat) + self.mon_scalar_proj(mscal)  # [B,12,d]
        # global token
        gc = torch.cat([self.weather(gcat[:, 0]), self.terrain(gcat[:, 1])], -1)
        glob = (self.global_cat_proj(gc) + self.global_scalar_proj(gscal))  # [B,d]
        # assemble the 16 tokens
        intent = self.intent.unsqueeze(0).expand(b, -1, -1)
        cls = self.cls.unsqueeze(0).expand(b, -1, -1)
        tokens = torch.cat([glob.unsqueeze(1), mon, intent, cls], 1)  # [B,16,d]
        tokens = tokens + self.role(self.roles).unsqueeze(0)
        return self.norm(tokens)


class RoleTypedAttention(nn.Module):
    """Multi-head attention with per-role Q/K/V/O and a damage-edge bias."""

    def __init__(self, jcfg):
        """Allocate role-indexed projection weights for ``N_ROLES`` roles."""
        super().__init__()
        d, h = jcfg.d_model, jcfg.n_heads
        assert d % h == 0
        self.d, self.h, self.dh = d, h, d // h
        self.wq = nn.Parameter(torch.empty(N_ROLES, d, d))
        self.wk = nn.Parameter(torch.empty(N_ROLES, d, d))
        self.wv = nn.Parameter(torch.empty(N_ROLES, d, d))
        self.wo = nn.Parameter(torch.empty(N_ROLES, d, d))
        for w in (self.wq, self.wk, self.wv, self.wo):
            nn.init.xavier_uniform_(w)
        self.register_buffer("roles", torch.tensor(ROLES), persistent=False)

    def forward(self, x, bias):
        """Attend over ``x`` ([B,16,d]); ``bias`` is additive ``[B,H,16,16]``."""
        b, n, d = x.shape
        wq, wk, wv = (w[self.roles] for w in (self.wq, self.wk, self.wv))  # [N,d,d]
        q = torch.einsum("bnd,nde->bne", x, wq)
        k = torch.einsum("bnd,nde->bne", x, wk)
        v = torch.einsum("bnd,nde->bne", x, wv)
        q, k, v = (t.view(b, n, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.dh ** 0.5)
        if bias is not None:
            scores = scores + bias
        out = torch.matmul(scores.softmax(-1), v)          # [B,H,N,dh]
        out = out.transpose(1, 2).reshape(b, n, d)
        return torch.einsum("bnd,nde->bne", out, self.wo[self.roles])


class RoleTypedLayer(nn.Module):
    """Pre-norm transformer block using role-typed attention + shared FFN."""

    def __init__(self, jcfg):
        """Build the attention, norms, and shared feed-forward network."""
        super().__init__()
        d = jcfg.d_model
        self.n1 = nn.LayerNorm(d)
        self.attn = RoleTypedAttention(jcfg)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, jcfg.d_ff), nn.GELU(),
                                nn.Dropout(jcfg.dropout), nn.Linear(jcfg.d_ff, d))
        self.drop = nn.Dropout(jcfg.dropout)

    def forward(self, x, bias):
        """Apply attention and FFN sublayers with residual connections."""
        x = x + self.drop(self.attn(self.n1(x), bias))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class DamageBias(nn.Module):
    """Turn an ally->foe damage edge matrix into per-head attention bias."""

    def __init__(self, jcfg):
        """Create the scalar-edge -> per-head bias MLP."""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, jcfg.n_heads), nn.Tanh(),
                                 nn.Linear(jcfg.n_heads, jcfg.n_heads))

    def forward(self, dmg_edge):
        """``[B,6,6]`` ally->foe damage -> additive bias ``[B,H,16,16]``."""
        b = dmg_edge.shape[0]
        full = dmg_edge.new_zeros(b, N_TOKENS, N_TOKENS)
        full[:, ALLY0:ALLY0 + N_MON, FOE0:FOE0 + N_MON] = dmg_edge
        full[:, FOE0:FOE0 + N_MON, ALLY0:ALLY0 + N_MON] = dmg_edge.transpose(1, 2)
        bias = self.mlp(full.unsqueeze(-1))                 # [B,16,16,H]
        return bias.permute(0, 3, 1, 2)


class Encoder(nn.Module):
    """Embedder + role-typed transformer stack -> context latents Z."""

    def __init__(self, sizes, jcfg):
        """Build the embedder, damage-bias module, and encoder layers."""
        super().__init__()
        self.embedder = Embedder(sizes, jcfg)
        self.dmg_bias = DamageBias(jcfg)
        self.layers = nn.ModuleList(RoleTypedLayer(jcfg)
                                    for _ in range(jcfg.n_enc_layers))

    def forward(self, gcat, gscal, mcat, mscal, dmg_edge):
        """Encode one batch of positions to ``[B, 16, d]`` latents."""
        x = self.embedder(gcat, gscal, mcat, mscal)
        bias = self.dmg_bias(dmg_edge)
        for layer in self.layers:
            x = layer(x, bias)
        return x


class Predictor(nn.Module):
    """Action-conditioned dynamics: (Z, joint action) -> next latents Ẑ'."""

    def __init__(self, emb, jcfg):
        """Build action tables (sharing the move table) and predictor layers."""
        super().__init__()
        self.emb = [emb]        # reference, not a submodule (weights shared)
        de, d = jcfg.d_embed, jcfg.d_model
        self.kind = nn.Embedding(3, de)
        self.target = nn.Embedding(4, de)
        self.switch = nn.Embedding(N_MON + 1, de)
        self.act_proj = nn.Linear(4 * de + 3, d)
        self.dmg_bias = DamageBias(jcfg)
        self.layers = nn.ModuleList(RoleTypedLayer(jcfg)
                                    for _ in range(jcfg.n_pred_layers))
        self.role = nn.Embedding(N_ROLES, d)
        self.register_buffer("roles", torch.tensor(ROLES), persistent=False)

    def action_embed(self, act):
        """``[B,12,7]`` action arrays -> ``[B,12,d]`` action embeddings."""
        move_tab = self.emb[0].move
        cat = torch.cat([self.kind(act[..., 0]), move_tab(act[..., 1]),
                         self.target(act[..., 2]), self.switch(act[..., 4])], -1)
        scal = torch.stack([act[..., 3].float(), act[..., 5].float() / 250.0,
                            act[..., 6].float() / 12.0], -1)
        return self.act_proj(torch.cat([cat, scal], -1))

    def forward(self, z, act, dmg_edge):
        """Predict next-state latents from context Z and a joint action."""
        x = z + self.role(self.roles).unsqueeze(0)
        a = self.action_embed(act)                          # [B,12,d]
        x = x.clone()
        x[:, ALLY0:ALLY0 + MON_TOKENS] = x[:, ALLY0:ALLY0 + MON_TOKENS] + a
        bias = self.dmg_bias(dmg_edge)
        for layer in self.layers:
            x = layer(x, bias)
        return x


class JEPAWorldModel(nn.Module):
    """Full world model: online/target encoders, predictor, and readout heads."""

    def __init__(self, sizes, jcfg, vocab_state=None):
        """Assemble encoders, predictor, and heads for vocab ``sizes``."""
        super().__init__()
        self.jcfg = jcfg
        self.sizes = dict(sizes)
        self.vocab_state = vocab_state
        d = jcfg.d_model
        self.encoder = Encoder(sizes, jcfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(self.encoder.embedder, jcfg)
        # readouts
        self.value_head = nn.Linear(d, 1)
        self.hp_head = nn.Linear(d, 1)
        self.faint_head = nn.Linear(d, 1)
        self.status_head = nn.Linear(d, len(STATUSES))
        self.weather_head = nn.Linear(d, len(WEATHERS))
        self.terrain_head = nn.Linear(d, len(TERRAINS))
        self.tr_head = nn.Linear(d, 1)
        self.screen_head = nn.Linear(d, 8)          # my/opp x tw/ref/ls/av
        self.order_head = nn.Linear(d, 3)
        self.my_prior = nn.Linear(d, 2 * N_SLOT_ACTIONS)
        self.opp_policy = nn.Linear(d, 2 * N_SLOT_ACTIONS)

    # -- encode/predict ------------------------------------------------------
    def encode(self, pos):
        """Encode a position dict of tensors with the online encoder -> Z."""
        return self.encoder(pos["gcat"], pos["gscal"], pos["mcat"],
                            pos["mscal"], pos["dmg"])

    @torch.no_grad()
    def target_encode(self, pos):
        """Encode a position with the EMA target encoder (stop-grad)."""
        return self.target_encoder(pos["gcat"], pos["gscal"], pos["mcat"],
                                   pos["mscal"], pos["dmg"])

    def predict(self, z, act, dmg):
        """Run the dynamics predictor for one joint action -> Ẑ'."""
        return self.predictor(z, act, dmg)

    def value(self, zp):
        """Read the searching-side win-probability value off ``Ẑ'`` (CLS)."""
        return torch.tanh(self.value_head(zp[:, CLS_TOK])).squeeze(-1)

    def policies(self, z):
        """Return ``(my_prior, opp_policy)`` slot logits ``[B, 2, 39]`` from Z."""
        cls, intent = z[:, CLS_TOK], z[:, 13:15].mean(1)
        my = self.my_prior(cls).view(-1, 2, N_SLOT_ACTIONS)
        opp = self.opp_policy(intent).view(-1, 2, N_SLOT_ACTIONS)
        return my, opp

    def grounded(self, zp):
        """Grounded next-state decoders read off predicted latents ``Ẑ'``."""
        mon = zp[:, ALLY0:ALLY0 + MON_TOKENS]
        return {
            "hp": torch.sigmoid(self.hp_head(mon)).squeeze(-1),      # [B,12]
            "faint": self.faint_head(mon).squeeze(-1),               # [B,12]
            "status": self.status_head(mon),                         # [B,12,S]
            "weather": self.weather_head(zp[:, GLOBAL_TOK]),
            "terrain": self.terrain_head(zp[:, GLOBAL_TOK]),
            "tr": self.tr_head(zp[:, GLOBAL_TOK]).squeeze(-1),
            "screens": self.screen_head(zp[:, GLOBAL_TOK]),
            "order": self.order_head(zp[:, CLS_TOK]),
        }

    @torch.no_grad()
    def update_ema(self):
        """EMA-update the target encoder toward the online encoder weights."""
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
                    "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a checkpoint, returning ``(model, vocab_state_dict_or_None)``."""
        from jepa.config import JEPAConfig
        ck = torch.load(path, map_location=device, weights_only=False)
        jcfg = JEPAConfig(**{k: v for k, v in ck["jcfg"].items()
                             if k in JEPAConfig.__dataclass_fields__})
        m = cls(ck["sizes"], jcfg, ck.get("vocab_state")).to(device)
        m.load_state_dict(ck["state"])
        return m, ck.get("vocab_state")
