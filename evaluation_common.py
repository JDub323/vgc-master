"""Shared loading/inference helpers for offline policy evaluation tools."""

import json
from pathlib import Path

import numpy as np
import torch

from actions import (N_JOINT_ACTIONS, SlotAction, T_ALLY, T_AUTO, T_FOE_A,
                     T_FOE_B, from_index, joint_index, joint_ok)
from config import CFG
from models.policy_value import PolicyValueNet
from train import Shards


def target_codes(dex_target):
    """Legal target encodings for a move's dex target class.

    The single source of truth for target legality in offline evaluation;
    mirrors the request-driven branch in ``actions.legal_slot_actions``."""
    if dex_target in ("normal", "any", "adjacentFoe"):
        return (T_FOE_A, T_FOE_B, T_ALLY)
    if dex_target == "adjacentAlly":
        return (T_ALLY,)
    return (T_AUTO,)


def load_test_predictions(checkpoint, cfg=CFG):
    """Load the test split, damage grids, model, joint predictions, values."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Shards("test", cfg)
    files = sorted(Path(cfg.prepped_dir).glob("test_*.npz"))
    if not files:
        raise FileNotFoundError(f"no test shards under {cfg.prepped_dir}")
    dmg_active = np.concatenate([np.load(f)["dmg_active"] for f in files])
    # exp/entity-hybrid: checkpoints record their architecture; entity
    # checkpoints load as EntityHybridNet, everything else as PolicyValueNet
    from models.entity_hybrid import load_any_policy_model
    model = load_any_policy_model(checkpoint, cfg, device)
    dists, values = [], []
    for i in range(0, len(ds), cfg.batch_size):
        dist, value, _ = model.predict_batch(ds.tokens[i:i + cfg.batch_size])
        dists.append(dist)
        values.append(value)
    return (ds, dmg_active, model,
            np.concatenate(dists).astype(np.float64),
            np.concatenate(values).astype(np.float64))


class PositionLegality:
    """Reconstruct per-position legal joint actions from tokenized states.

    The search never ranks over all 39x39 actions: ``PolicyValuePrior``
    normalizes over the actions the sim request says are legal *here* and only
    then takes the top-k. Offline evaluation has no request — prepped shards
    keep tokens and labels, not legality — so scoring against the static mask
    alone lets high-prior illegal actions push the human's action down the
    ranking and understates what search actually sees.

    This rebuilds the legal set from the tokens, mirroring
    ``actions.legal_slot_actions`` on everything the tokens retain: which mon
    holds each slot, which move slots are filled, each move's target class
    (from ``dex.json``), which team members can be switched to, and whether a
    mega is still available to a mon holding its own stone.

    Four things the tokens do not retain, all resolved permissively (the same
    benefit-of-the-doubt ``DamageStatusSwitchCandidates`` gives its floor):
    disabled moves, zero-PP moves, trapping, and which four mons were brought.
    The result is therefore a *superset* of the true legal set, so metrics
    computed against it are a lower bound on the true position-legal number —
    but a far tighter one than the ~1521-action static mask.
    """

    def __init__(self, tokenizer, dex_path):
        """Load move/item metadata for target classes and mega stones."""
        self.tok = tokenizer
        dex = json.loads(Path(dex_path).read_text())
        self.moves = dex["moves"]
        self.items = dex["items"]

    def _can_mega(self, block):
        """Return whether this mon holds a stone that megas its own species."""
        item, species = block[2], block[1]
        if not item.startswith("item:") or not species.startswith("species:"):
            return False
        stone = self.items.get(item[5:], {}).get("megaStone")
        return bool(stone) and species[8:] in stone

    def _move_target(self, block, move_slot):
        """Return the dex target class of a filled move slot, else ``None``."""
        token = block[6 + move_slot]
        if not token.startswith("move:"):
            return None
        return self.moves.get(token[5:], {}).get("target", "normal")

    def _slot_actions(self, blocks, mega_available, slot):
        """Return the legal ``SlotAction`` list for one active slot."""
        block = self._active(blocks, slot)
        if block is None:            # empty slot, or a fainted mon holds it
            return [SlotAction("pass")]
        actions = []
        for k, b in enumerate(blocks):
            if b[0] in ("BENCH", "UNSEEN") and b[4] != "HP_0":
                actions.append(SlotAction("switch", switch_to=k))
        can_mega = mega_available and self._can_mega(block)
        for move_slot in range(4):
            target = self._move_target(block, move_slot)
            if target is None:
                continue
            for code in target_codes(target):
                actions.append(
                    SlotAction("move", move_slot=move_slot, target=code))
                if can_mega:
                    actions.append(SlotAction("move", move_slot=move_slot,
                                              target=code, mega=True))
        return actions or [SlotAction("pass")]

    @staticmethod
    def _active(blocks, slot):
        """Return the block holding ``slot``, or ``None``."""
        want = "SLOT_A" if slot == 0 else "SLOT_B"
        return next((b for b in blocks if b[0] == want), None)

    def _decode(self, ids):
        """Return (my six mon blocks, mega-available flag) for one row."""
        tok = self.tok
        names = tok.decode(ids[:tok.opp_base])       # globals + my six blocks
        return ([names[tok.my_base + k * tok.mon_block:
                       tok.my_base + (k + 1) * tok.mon_block]
                 for k in range(6)], names[5] == "MEGA_AVAIL")

    def _label_actions(self, blocks, act, slot):
        """Return the slot actions consistent with one recorded label.

        Usually the label itself. But ``data.py`` infers a move's target from
        the protocol line, and that inference does not agree with the action
        space the search enumerates:

          * a spread move (``allAdjacentFoes``) is recorded with a specific
            target whenever Showdown omits its ``[spread]`` tag, which it does
            when only one target was hit;
          * a single-target (``normal``) move is recorded as ``T_AUTO`` — the
            parser's default — whenever its protocol line carries no target
            reference, e.g. when the move fizzled.

        Both encode a (move, target) pair ``legal_slot_actions`` can never
        propose, so the model is trained on an index the search never reads.
        Rather than score that as a model failure, project such a label onto
        the target codes the move actually admits: a set of candidate actions,
        collapsing to the exact label whenever the label is already legal."""
        if act.kind != "move":
            return [act], False
        block = self._active(blocks, slot)
        if block is None:
            return [act], False
        target = self._move_target(block, act.move_slot)
        if target is None:
            return [act], False
        codes = target_codes(target)
        if act.target in codes:
            return [act], False
        return ([SlotAction("move", move_slot=act.move_slot, target=code,
                            mega=act.mega) for code in codes], True)

    def mask(self, tokens):
        """Return a ``bool[B, N_JOINT_ACTIONS]`` position-legal action mask."""
        out = np.zeros((len(tokens), N_JOINT_ACTIONS), dtype=bool)
        for i, ids in enumerate(tokens):
            blocks, mega_available = self._decode(ids)
            a_actions = self._slot_actions(blocks, mega_available, 0)
            b_actions = self._slot_actions(blocks, mega_available, 1)
            for a in a_actions:
                for b in b_actions:
                    if joint_ok(a, b):
                        out[i, joint_index(a, b)] = True
        return out

    def label_mask(self, tokens, acts):
        """Return (bool[B, N_JOINT_ACTIONS] label sets, projected-row count).

        Each row marks every joint action consistent with the recorded human
        label — one action for a well-formed label, several when the label's
        target had to be projected (see ``_label_actions``)."""
        out = np.zeros((len(tokens), N_JOINT_ACTIONS), dtype=bool)
        projected = 0
        for i, ids in enumerate(tokens):
            blocks, _ = self._decode(ids)
            a_cands, a_proj = self._label_actions(
                blocks, from_index(int(acts[i, 0])), 0)
            b_cands, b_proj = self._label_actions(
                blocks, from_index(int(acts[i, 1])), 1)
            projected += a_proj or b_proj
            for a in a_cands:
                for b in b_cands:
                    if joint_ok(a, b):
                        out[i, joint_index(a, b)] = True
        return out, projected


def apply_legal_mask(flat, legal):
    """Renormalize joint distributions over legal actions, as search does.

    Mirrors ``agents/priors/v1.py``: keep legal mass and renormalize; a row
    whose legal mass is zero falls back to uniform over its legal actions."""
    masked = flat * legal
    total = masked.sum(axis=1, keepdims=True)
    counts = np.maximum(legal.sum(axis=1, keepdims=True), 1)
    uniform = legal / counts
    return np.where(total > 0, masked / np.maximum(total, 1e-300), uniform)
