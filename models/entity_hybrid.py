"""EntityHybridNet: entity-level transformer + MLP hybrid over layout 3.

The flat baseline attends over all 561 raw tokens, spending most of its
attention pairs on relationships the fixed layout already encodes ("this
token is mon 6's third boost"). An MLP treats every input position uniquely
and is therefore fast but brittle to team/move reordering. This model splits
the difference:

  561 layout-3 tokens (damage block [273:561] DROPPED — the ablation showed
  it carries no supervised policy signal in raw-token form)
     -> shared token embeddings
     -> shared move encoder (move token + move-slot embedding), all 48 moves
     -> shared Pokemon MLP over each mon's 18-token block (moves replaced by
        their encoded vectors); opponents additionally fold in their 7-token
        belief block
     -> + side / team-slot / entity-type embeddings (kept deliberately: team
        index IS action semantics — switch action k targets preview slot k)
     -> a small transformer over 13 entities (1 global + 6 mine + 6 theirs)
     -> flatten in fixed order -> residual MLP trunk
     -> the exact same heads as PolicyValueNet: 1521-way joint policy over
        the static mask, tanh value, per-opponent-mon item/ability/move aux.

Weight sharing across mons/moves gives transformer-style generalization
under reordering; the fixed-order flatten + MLP trunk gives the MLP's direct
position-to-action wiring. Attention cost falls from 561^2 to 13^2 pairs.

predict_batch / save / load mirror PolicyValueNet's contracts, so the model
drops into the DUCT chooser, the BatchedEvaluator, and evaluate.py unchanged.
Checkpoints carry hp["arch"] = "entity_hybrid" so loaders can dispatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from actions import N_JOINT_ACTIONS, static_joint_mask
from config import CFG, config_from_snapshot, config_snapshot
from tokenizer import BELIEF_BLOCK, MON_BLOCK, N_MONS

# layout-3 geometry, mirrored from PositionTokenizer.__init__ (asserted at
# construction so a layout bump fails loudly here instead of training garbage)
MON_T = MON_BLOCK + 1            # 18 tokens per mon block
BEL_T = BELIEF_BLOCK + 2         # 7 tokens per opponent belief block
GLOBAL_T = 15                    # CLS + field + both sides' conditions
MY_BASE = GLOBAL_T
OPP_BASE = MY_BASE + N_MONS * MON_T
BELIEF_BASE = OPP_BASE + N_MONS * MON_T
DMG_BASE = BELIEF_BASE + N_MONS * BEL_T          # 273; everything after is dropped
MOVE_LO, MOVE_HI = 6, 10         # move-token positions inside a mon block
N_ENTITIES = 1 + 2 * N_MONS      # global + my 6 + opp 6

ENTITY_MODEL_CFG_FIELDS = ("d_token", "d_entity", "mon_hidden", "n_ent_layers",
                           "n_ent_heads", "ent_ff", "d_trunk",
                           "n_trunk_blocks", "dropout")
ENTITY_MODEL_CFG_DEFAULTS = {
    "d_token": 64, "d_entity": 256, "mon_hidden": 384, "n_ent_layers": 2,
    "n_ent_heads": 8, "ent_ff": 1024, "d_trunk": 448, "n_trunk_blocks": 2,
    "dropout": 0.1,
}


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: x + W2 gelu(W1 LN(x))."""

    def __init__(self, d, dropout):
        """Build the norm and the two square linear layers."""
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """Return the residual-updated activations."""
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))


class EntityHybridNet(nn.Module):
    """Entity-transformer/MLP hybrid joint-policy/value/aux network."""

    def __init__(self, vocab_size, n_tokens, opp_positions,
                 n_moves, n_items, n_abilities, cfg=CFG, policy_head="joint",
                 model_cfg=None):
        """Constructor signature mirrors PolicyValueNet so call sites swap
        freely. opp_positions is accepted for compatibility but unused: aux
        heads read the six contextualized opponent entity vectors instead of
        raw token positions."""
        super().__init__()
        if policy_head != "joint":
            raise ValueError("only the joint policy head is supported")
        assert n_tokens == DMG_BASE + N_MONS * 4 * N_MONS * 2, \
            f"layout drift: expected 561 layout-3 tokens, got {n_tokens}"
        model_cfg = dict(ENTITY_MODEL_CFG_DEFAULTS) | dict(model_cfg or {})
        self.cfg_snapshot = config_snapshot(cfg)
        self.hp = {"vocab_size": vocab_size, "n_tokens": n_tokens,
                   "opp_positions": list(opp_positions), "n_moves": n_moves,
                   "n_items": n_items, "n_abilities": n_abilities,
                   "policy_head": policy_head, "arch": "entity_hybrid",
                   "model_cfg": dict(model_cfg)}
        self.policy_head = policy_head
        dt, de = int(model_cfg["d_token"]), int(model_cfg["d_entity"])
        drop = float(model_cfg["dropout"])

        self.emb = nn.Embedding(vocab_size, dt)
        # shared move encoder: token embedding + move-slot embedding -> MLP.
        # Slot identity is retained (the action label is slot-indexed) but the
        # move's meaning is learned once, not once per slot per mon.
        self.move_slot_emb = nn.Parameter(torch.zeros(1, 1, 4, dt))
        self.move_mlp = nn.Sequential(nn.Linear(dt, dt), nn.GELU(),
                                      nn.Linear(dt, dt))
        # shared Pokemon encoder over the 18-position block (moves replaced by
        # their encoded vectors) — one set of weights for all 12 mons
        mh = int(model_cfg["mon_hidden"])
        self.mon_mlp = nn.Sequential(nn.Linear(MON_T * dt, mh), nn.GELU(),
                                     nn.Dropout(drop), nn.Linear(mh, de))
        # opponent-only belief projection, added to the opponent entity vector
        self.belief_proj = nn.Linear(BEL_T * dt, de)
        self.global_proj = nn.Linear(GLOBAL_T * dt, de)
        # entity identity: [global, my 0..5, opp 0..5] — this is the model's
        # only positional signal, and it is per-entity, not per-token
        self.entity_emb = nn.Parameter(torch.zeros(1, N_ENTITIES, de))

        layer = nn.TransformerEncoderLayer(
            de, int(model_cfg["n_ent_heads"]), int(model_cfg["ent_ff"]),
            drop, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer,
                                             int(model_cfg["n_ent_layers"]))
        self.norm = nn.LayerNorm(de)

        dtr = int(model_cfg["d_trunk"])
        self.trunk_in = nn.Linear(N_ENTITIES * de, dtr)
        self.trunk = nn.Sequential(*[ResidualBlock(dtr, drop)
                                     for _ in range(int(model_cfg["n_trunk_blocks"]))])
        self.trunk_norm = nn.LayerNorm(dtr)
        self.joint_head = nn.Linear(dtr, N_JOINT_ACTIONS)
        self.value_head = nn.Linear(dtr, 1)
        self.item_head = nn.Linear(de, n_items + 1)
        self.ability_head = nn.Linear(de, n_abilities + 1)
        self.moves_head = nn.Linear(de, n_moves + 1)
        self.register_buffer("joint_mask",
                             torch.from_numpy(static_joint_mask().reshape(-1)),
                             persistent=False)

    def _entities(self, tokens):
        """Tokens [B, 561] -> entity vectors [B, 13, d_entity]."""
        B = tokens.shape[0]
        e = self.emb(tokens[:, :DMG_BASE])              # damage block dropped
        g = self.global_proj(e[:, :GLOBAL_T].reshape(B, -1))
        mons = e[:, MY_BASE:BELIEF_BASE].reshape(B, 2 * N_MONS, MON_T, -1)
        mv = self.move_mlp(mons[:, :, MOVE_LO:MOVE_HI] + self.move_slot_emb)
        mons = torch.cat([mons[:, :, :MOVE_LO], mv, mons[:, :, MOVE_HI:]], 2)
        ent = self.mon_mlp(mons.reshape(B, 2 * N_MONS, -1))
        bel = e[:, BELIEF_BASE:DMG_BASE].reshape(B, N_MONS, -1)
        ent = torch.cat([ent[:, :N_MONS],
                         ent[:, N_MONS:] + self.belief_proj(bel)], 1)
        return torch.cat([g.unsqueeze(1), ent], 1) + self.entity_emb

    def forward(self, tokens):
        """Return joint-policy logits, scalar value, and set predictions."""
        h = self.norm(self.encoder(self._entities(tokens)))
        t = self.trunk_norm(self.trunk(F.gelu(
            self.trunk_in(h.flatten(1)))))
        pol = self.joint_head(t)
        value = torch.tanh(self.value_head(t)).squeeze(-1)
        opp_h = h[:, 1 + N_MONS:]                        # [B, 6, d_entity]
        aux = (self.item_head(opp_h), self.ability_head(opp_h),
               self.moves_head(opp_h))
        return pol, value, aux

    def joint_dist(self, pol):
        """Normalize logits over statically legal joint actions."""
        return F.softmax(pol.masked_fill(~self.joint_mask, float("-inf")), -1)

    @torch.no_grad()
    def predict_batch(self, tokens):
        """Same contract as PolicyValueNet.predict_batch: numpy joint dists,
        values, and set predictions for use inside search."""
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
        from models.policy_value import clean_state_dict
        torch.save({"hp": self.hp, "state": clean_state_dict(self),
                    "cfg": self.cfg_snapshot}, path)

    @classmethod
    def load(cls, path, cfg=CFG, device="cpu"):
        """Return a checkpoint-restored model on ``device``."""
        from models.policy_value import strip_compile_prefix
        ck = torch.load(path, map_location=device, weights_only=False)
        load_cfg = config_from_snapshot(ck.get("cfg"), base=cfg)
        hp = dict(ck["hp"])
        hp.pop("arch", None)
        m = cls(**hp, cfg=load_cfg).to(device)
        m.load_state_dict(strip_compile_prefix(ck["state"]))
        return m


def is_entity_checkpoint(path):
    """True when the checkpoint at ``path`` was saved by EntityHybridNet."""
    import torch as _torch
    ck = _torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("hp", {}).get("arch") == "entity_hybrid"


def load_any_policy_model(path, cfg=CFG, device="cpu"):
    """Load a checkpoint as EntityHybridNet or PolicyValueNet by its
    recorded architecture — the one loader every entry point can call."""
    if is_entity_checkpoint(path):
        return EntityHybridNet.load(path, cfg, device)
    from models.policy_value import PolicyValueNet
    return PolicyValueNet.load(path, cfg, device)
