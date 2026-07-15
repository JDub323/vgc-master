"""Seq2SeqPointerNet: legal-move pointer decoder over the entity encoder.

The baseline (and the entity hybrid) score all 1521 joint actions with a
fixed output head and rely on masking to hide the illegal ones. This model
inverts that: the *legal actions themselves* are the decoder's input. Each of
the 39 per-slot action indices becomes a candidate token built from

  action-index embedding            (which abstract action this is)
  + content embedding               (move actions: the active mon's move
                                     token; switches: the target mon's
                                     species token — gathered from the input)
  + target-code + mega embeddings   (move variants share content, differ here)
  + slot embedding                  (slot A vs slot B)

with illegal candidates masked out. One transformer decoder layer runs
self-attention over the 78 candidates (both slots) and cross-attention to the
13 entity vectors of the entity-hybrid encoder (reused unchanged: token emb
-> shared move encoder -> shared per-mon MLP -> 13 entities -> 2 transformer
layers; only ``tokens[:, :273]`` is read, the damage suffix is dropped).

Heads:
  slot A   pointer head, 39 logits, legal-masked
  slot B|A bilinear pairwise head [39, 39], masked by legal-B AND the static
           ``joint_ok`` grid (chain rule: P(a, b) = P(a) * P(b | a))
  value    tanh scalar off the entity-hybrid's flatten + residual-MLP trunk
  aux      per-opponent-mon item / ability / move heads (as EntityHybridNet)

Legality is NOT in the tokens: it is reconstructed from them through
``evaluation_common.PositionLegality`` — the same permissive superset at
train time (``seq2seq_prep.py`` sidecars) and play time (``predict_batch``
computes it per call). ``predict_batch`` chain-rule-expands to the flat
[1521] joint grid and renormalizes, so DUCT search, evaluate.py and
agent_server all consume it unchanged. Checkpoints carry
``hp["arch"] = "seq2seq_pointer"``; ``load_any_policy_model`` dispatches
seq2seq / entity-hybrid / PolicyValueNet checkpoints by that tag.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from actions import (N_JOINT_ACTIONS, N_SLOT_ACTIONS, from_index,
                     static_joint_mask, to_index)
from config import CFG, config_from_snapshot, config_snapshot
from models.entity_hybrid import (BEL_T, BELIEF_BASE, DMG_BASE, GLOBAL_T,
                                  MON_T, MOVE_HI, MOVE_LO, MY_BASE,
                                  N_ENTITIES, ResidualBlock)
from tokenizer import N_MONS

SEQ2SEQ_MODEL_CFG_FIELDS = ("d_token", "d_entity", "mon_hidden",
                            "n_ent_layers", "n_ent_heads", "ent_ff",
                            "dec_heads", "dec_ff", "d_trunk",
                            "n_trunk_blocks", "dropout")
SEQ2SEQ_MODEL_CFG_DEFAULTS = {
    "d_token": 64, "d_entity": 256, "mon_hidden": 384, "n_ent_layers": 2,
    "n_ent_heads": 8, "ent_ff": 1024, "dec_heads": 8, "dec_ff": 1024,
    "d_trunk": 448, "n_trunk_blocks": 1, "dropout": 0.1,
}

# candidate-table codes for the non-move slots of the target/mega embeddings
TGT_NA, MEGA_NA = 4, 2


def _candidate_tables():
    """Static per-action-index lookup rows for the 39 slot actions.

    Returns (kind, move_slot, target, mega) int64 arrays of length 39; kind is
    0 pass / 1 move / 2 switch, non-move rows read the *_NA embedding slots,
    and move_slot doubles as the switch target's team index for kind 2 (it is
    only ever used through kind-specific gathers)."""
    kind = np.zeros(N_SLOT_ACTIONS, np.int64)
    mslot = np.zeros(N_SLOT_ACTIONS, np.int64)
    target = np.full(N_SLOT_ACTIONS, TGT_NA, np.int64)
    mega = np.full(N_SLOT_ACTIONS, MEGA_NA, np.int64)
    for i in range(N_SLOT_ACTIONS):
        a = from_index(i)
        if a.kind == "move":
            kind[i], mslot[i] = 1, a.move_slot
            target[i], mega[i] = a.target, int(a.mega)
        elif a.kind == "switch":
            kind[i], mslot[i] = 2, a.switch_to
    return kind, mslot, target, mega


def slot_legal_masks(legality, tokens):
    """Per-slot legal masks ``(legal_a, legal_b)``, each ``bool[B, 39]``.

    Reconstructed from tokens via ``PositionLegality._slot_actions`` — the
    permissive superset (disabled/zero-PP moves, trapping, and the bring-four
    are unknowable from tokens). The single legality source shared by
    training (seq2seq_prep.py) and play (predict_batch)."""
    la = np.zeros((len(tokens), N_SLOT_ACTIONS), dtype=bool)
    lb = np.zeros_like(la)
    for i, ids in enumerate(tokens):
        blocks, mega_available = legality._decode(ids)
        for a in legality._slot_actions(blocks, mega_available, 0):
            la[i, to_index(a)] = True
        for b in legality._slot_actions(blocks, mega_available, 1):
            lb[i, to_index(b)] = True
    return la, lb


def slot_label_sets(legality, tokens, acts):
    """Projected per-slot label sets ``(set_a, set_b, n_projected)``.

    Each row marks the slot actions consistent with the recorded human label
    via ``PositionLegality._label_actions``: exactly the label when it is
    well-formed, or its projection onto the move's real target codes for the
    8.7% of labels whose recorded target the legal set never contains
    (KNOWN_ISSUES.md #3). ``n_projected`` counts rows where either slot was
    projected."""
    sa = np.zeros((len(tokens), N_SLOT_ACTIONS), dtype=bool)
    sb = np.zeros_like(sa)
    projected = 0
    for i, ids in enumerate(tokens):
        blocks, _ = legality._decode(ids)
        a_cands, a_proj = legality._label_actions(
            blocks, from_index(int(acts[i, 0])), 0)
        b_cands, b_proj = legality._label_actions(
            blocks, from_index(int(acts[i, 1])), 1)
        projected += bool(a_proj or b_proj)
        for a in a_cands:
            sa[i, to_index(a)] = True
        for b in b_cands:
            sb[i, to_index(b)] = True
    return sa, sb, projected


class Seq2SeqPointerNet(nn.Module):
    """Legal-action pointer policy + value/aux net (module docstring)."""

    def __init__(self, vocab_size, n_tokens, opp_positions, n_moves, n_items,
                 n_abilities, cfg=CFG, policy_head="joint", model_cfg=None,
                 slot_token_ids=None):
        """Constructor signature mirrors PolicyValueNet plus
        ``slot_token_ids = (vocab["SLOT_A"], vocab["SLOT_B"])``, needed to
        locate each slot's active mon for candidate content gathering.
        opp_positions is accepted for compatibility but unused (aux heads
        read the contextualized opponent entity vectors)."""
        super().__init__()
        if policy_head != "joint":
            raise ValueError("only the joint policy head is supported")
        if slot_token_ids is None:
            raise ValueError("slot_token_ids=(SLOT_A id, SLOT_B id) required")
        assert n_tokens == DMG_BASE + N_MONS * 4 * N_MONS * 2, \
            f"layout drift: expected 561 layout-3 tokens, got {n_tokens}"
        model_cfg = dict(SEQ2SEQ_MODEL_CFG_DEFAULTS) | dict(model_cfg or {})
        self.cfg_snapshot = config_snapshot(cfg)
        self._runtime_cfg = cfg
        self._legality = None
        self.hp = {"vocab_size": vocab_size, "n_tokens": n_tokens,
                   "opp_positions": list(opp_positions), "n_moves": n_moves,
                   "n_items": n_items, "n_abilities": n_abilities,
                   "policy_head": policy_head, "arch": "seq2seq_pointer",
                   "model_cfg": dict(model_cfg),
                   "slot_token_ids": list(slot_token_ids)}
        self.policy_head = policy_head
        self.slot_token_ids = tuple(int(t) for t in slot_token_ids)
        dt, de = int(model_cfg["d_token"]), int(model_cfg["d_entity"])
        drop = float(model_cfg["dropout"])

        # ---- encoder: identical stack to EntityHybridNet (fresh weights) ----
        self.emb = nn.Embedding(vocab_size, dt)
        self.move_slot_emb = nn.Parameter(torch.zeros(1, 1, 4, dt))
        self.move_mlp = nn.Sequential(nn.Linear(dt, dt), nn.GELU(),
                                      nn.Linear(dt, dt))
        mh = int(model_cfg["mon_hidden"])
        self.mon_mlp = nn.Sequential(nn.Linear(MON_T * dt, mh), nn.GELU(),
                                     nn.Dropout(drop), nn.Linear(mh, de))
        self.belief_proj = nn.Linear(BEL_T * dt, de)
        self.global_proj = nn.Linear(GLOBAL_T * dt, de)
        self.entity_emb = nn.Parameter(torch.zeros(1, N_ENTITIES, de))
        layer = nn.TransformerEncoderLayer(
            de, int(model_cfg["n_ent_heads"]), int(model_cfg["ent_ff"]),
            drop, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer,
                                             int(model_cfg["n_ent_layers"]))
        self.norm = nn.LayerNorm(de)

        # ---- candidate embeddings ----
        self.action_emb = nn.Embedding(N_SLOT_ACTIONS, de)
        self.slot_emb = nn.Parameter(torch.zeros(2, 1, de))
        self.target_emb = nn.Embedding(TGT_NA + 1, de)   # 4 codes + n/a
        self.mega_emb = nn.Embedding(MEGA_NA + 1, de)    # off/on + n/a
        self.content_proj = nn.Linear(dt, de)

        # ---- decoder: candidates attend to each other and to entities ----
        dec = nn.TransformerDecoderLayer(
            de, int(model_cfg["dec_heads"]), int(model_cfg["dec_ff"]),
            drop, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = dec
        self.dec_norm = nn.LayerNorm(de)
        self.ptr_a = nn.Linear(de, 1)
        self.pair_proj = nn.Linear(de, de, bias=False)
        self.pair_scale = float(de) ** 0.5

        # ---- value trunk (entity-hybrid's, minus the joint head) + aux ----
        dtr = int(model_cfg["d_trunk"])
        self.trunk_in = nn.Linear(N_ENTITIES * de, dtr)
        self.trunk = nn.Sequential(*[ResidualBlock(dtr, drop)
                                     for _ in range(int(model_cfg["n_trunk_blocks"]))])
        self.trunk_norm = nn.LayerNorm(dtr)
        self.value_head = nn.Linear(dtr, 1)
        self.item_head = nn.Linear(de, n_items + 1)
        self.ability_head = nn.Linear(de, n_abilities + 1)
        self.moves_head = nn.Linear(de, n_moves + 1)

        kind, mslot, target, mega = _candidate_tables()
        self.register_buffer("cand_kind", torch.from_numpy(kind),
                             persistent=False)
        self.register_buffer("cand_mslot", torch.from_numpy(mslot),
                             persistent=False)
        self.register_buffer("cand_target", torch.from_numpy(target),
                             persistent=False)
        self.register_buffer("cand_mega", torch.from_numpy(mega),
                             persistent=False)
        self.register_buffer("joint_ok_mask",
                             torch.from_numpy(static_joint_mask().copy()),
                             persistent=False)

    def _entities(self, tokens):
        """Tokens [B, 561] -> entity vectors [B, 13, d_entity] (the entity-
        hybrid encoder verbatim; the damage suffix is never read)."""
        B = tokens.shape[0]
        e = self.emb(tokens[:, :DMG_BASE])
        g = self.global_proj(e[:, :GLOBAL_T].reshape(B, -1))
        mons = e[:, MY_BASE:BELIEF_BASE].reshape(B, 2 * N_MONS, MON_T, -1)
        mv = self.move_mlp(mons[:, :, MOVE_LO:MOVE_HI] + self.move_slot_emb)
        mons = torch.cat([mons[:, :, :MOVE_LO], mv, mons[:, :, MOVE_HI:]], 2)
        ent = self.mon_mlp(mons.reshape(B, 2 * N_MONS, -1))
        bel = e[:, BELIEF_BASE:DMG_BASE].reshape(B, N_MONS, -1)
        ent = torch.cat([ent[:, :N_MONS],
                         ent[:, N_MONS:] + self.belief_proj(bel)], 1)
        return torch.cat([g.unsqueeze(1), ent], 1) + self.entity_emb

    def _candidates(self, tokens, slot):
        """Candidate vectors [B, 39, d_entity] for one slot.

        Content: move candidates read the acting mon's move token (all eight
        target/mega variants of a move share it — the variant identity rides
        the target/mega embeddings); switch candidates read the target team
        slot's species token; PASS carries no content. An empty slot (no mon
        holds it) zeroes the move content — its move candidates are illegal
        and masked anyway."""
        B = tokens.shape[0]
        blocks = tokens[:, MY_BASE:MY_BASE + N_MONS * MON_T]
        blocks = blocks.reshape(B, N_MONS, MON_T)
        active = blocks[:, :, 0] == self.slot_token_ids[slot]
        k = active.long().argmax(1)
        has_active = active.any(1)
        act_block = blocks[torch.arange(B, device=tokens.device), k]
        move_e = self.content_proj(self.emb(act_block[:, MOVE_LO:MOVE_HI]))
        move_e = move_e * has_active[:, None, None]
        species_e = self.content_proj(self.emb(blocks[:, :, 1]))
        content = torch.cat([move_e.new_zeros(B, 1, move_e.shape[-1]),
                             move_e[:, self.cand_mslot[1:33]],
                             species_e], 1)
        return (self.action_emb.weight.unsqueeze(0) + content
                + self.target_emb(self.cand_target).unsqueeze(0)
                + self.mega_emb(self.cand_mega).unsqueeze(0)
                + self.slot_emb[slot])

    def forward(self, tokens, legal_a, legal_b):
        """Return (log P(a) [B,39], log P(b|a) [B,39,39], value, aux).

        ``legal_a``/``legal_b`` are the per-slot bool masks from
        ``slot_legal_masks`` (training reads them from the prep sidecar,
        predict_batch computes them). Slot-A logits are masked to legal-A;
        the pairwise head is masked to legal-B AND ``joint_ok``; the rare
        (a-legal, no compatible b) row falls back to uniform over legal-B so
        no softmax ever sees an all-masked row."""
        h = self.norm(self.encoder(self._entities(tokens)))
        x = torch.cat([self._candidates(tokens, 0),
                       self._candidates(tokens, 1)], 1)
        pad = ~torch.cat([legal_a, legal_b], 1)
        d = self.dec_norm(self.decoder(x, h, tgt_key_padding_mask=pad))
        da, db = d[:, :N_SLOT_ACTIONS], d[:, N_SLOT_ACTIONS:]

        logit_a = self.ptr_a(da).squeeze(-1)
        log_pa = F.log_softmax(
            logit_a.masked_fill(~legal_a, float("-inf")), -1)
        pair = torch.einsum("bid,bjd->bij", self.pair_proj(da), db)
        pair = pair / self.pair_scale
        m = legal_b[:, None, :] & self.joint_ok_mask
        empty = ~m.any(-1, keepdim=True)
        m = m | (empty & legal_b[:, None, :])
        log_pb = F.log_softmax(pair.masked_fill(~m, float("-inf")), -1)

        t = self.trunk_norm(self.trunk(F.gelu(self.trunk_in(h.flatten(1)))))
        value = torch.tanh(self.value_head(t)).squeeze(-1)
        opp_h = h[:, 1 + N_MONS:]
        aux = (self.item_head(opp_h), self.ability_head(opp_h),
               self.moves_head(opp_h))
        return log_pa, log_pb, value, aux

    def _get_legality(self):
        """Lazily build the tokens->legal-set reconstructor (needs
        artifacts/vocab.json and artifacts/dex.json at play time)."""
        if self._legality is None:
            from evaluation_common import PositionLegality
            from tokenizer import PositionTokenizer
            cfg = self._runtime_cfg
            self._legality = PositionLegality(
                PositionTokenizer.load(cfg),
                cfg.artifacts_dir / "dex.json")
        return self._legality

    @torch.no_grad()
    def predict_batch(self, tokens):
        """Same contract as PolicyValueNet.predict_batch: numpy joint [B,1521]
        dists, values, and aux set predictions. The native (P(a), P(b|a))
        output is chain-rule expanded onto the flat joint grid and
        renormalized, so search/eval consume it unchanged."""
        self.eval()
        dev = next(self.parameters()).device
        tokens = np.asarray(tokens)
        la, lb = slot_legal_masks(self._get_legality(), tokens)
        t = torch.as_tensor(tokens, dtype=torch.long, device=dev)
        log_pa, log_pb, value, (items, abils, moves) = self(
            t, torch.from_numpy(la).to(dev), torch.from_numpy(lb).to(dev))
        joint = (log_pa.unsqueeze(-1) + log_pb).exp()
        joint = joint.reshape(len(tokens), N_JOINT_ACTIONS)
        joint = joint / joint.sum(-1, keepdim=True).clamp_min(1e-12)
        return (joint.cpu().numpy(), value.cpu().numpy(),
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


def checkpoint_arch(path):
    """Return the ``hp["arch"]`` tag of a checkpoint (None for baseline)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("hp", {}).get("arch")


def load_any_policy_model(path, cfg=CFG, device="cpu"):
    """Load a checkpoint as Seq2SeqPointerNet, EntityHybridNet, or
    PolicyValueNet by its recorded architecture — the one loader every
    branch-local entry point calls."""
    arch = checkpoint_arch(path)
    if arch == "seq2seq_pointer":
        return Seq2SeqPointerNet.load(path, cfg, device)
    if arch == "entity_hybrid":
        from models.entity_hybrid import EntityHybridNet
        return EntityHybridNet.load(path, cfg, device)
    from models.policy_value import PolicyValueNet
    return PolicyValueNet.load(path, cfg, device)
