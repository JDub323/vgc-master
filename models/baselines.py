"""Benchmark floors. Both expose predict_batch(dmg_active) -> per-slot action
distributions [B, 2, N_SLOT_ACTIONS] so evaluate.py scores them exactly like
the network. dmg_active is the prepped [B, my_slot, move, opp_slot] avg-damage%
grid (uint8)."""

import json
import random

import numpy as np

from actions import (N_SLOT_ACTIONS, SlotAction, T_ALLY, T_AUTO, T_FOE_A,
                     T_FOE_B, joint_ok, to_index)


class RandomPolicy:
    """Offline evaluator baseline with uniform per-slot distributions."""

    def predict_batch(self, dmg_active):
        """Return NumPy float ``[B,2,39]`` uniform distributions."""
        b = len(dmg_active)
        return np.full((b, 2, N_SLOT_ACTIONS), 1.0 / N_SLOT_ACTIONS)


class MaxDamagePolicy:
    """Picks, per slot, the (move, target) with the highest expected immediate
    damage. Slots with no damage information fall back to uniform."""

    def __init__(self, eps=0.05):
        """Store smoothing mass ``eps`` for non-greedy actions."""
        self.eps = eps   # smoothing so log-loss is finite

    def predict_batch(self, dmg_active):
        """Map NumPy damage grids ``[B,2,4,2]`` to ``[B,2,39]`` priors."""
        b = len(dmg_active)
        out = np.full((b, 2, N_SLOT_ACTIONS), self.eps / N_SLOT_ACTIONS)
        for i, grid in enumerate(np.asarray(dmg_active, dtype=np.float64)):
            for slot in range(2):
                g = grid[slot]                      # [move, opp_slot]
                if g.max() <= 0:
                    out[i, slot] = 1.0 / N_SLOT_ACTIONS
                    continue
                mv, tgt = np.unravel_index(g.argmax(), g.shape)
                a = SlotAction("move", move_slot=int(mv),
                               target=T_FOE_A if tgt == 0 else T_FOE_B)
                out[i, slot, to_index(a)] += 1.0 - self.eps
        return out


class DamageStatusSwitchCandidates:
    """Top-k sanity floor: four target-aware max-damage joints first, then a
    deterministic shuffle of status/max-damage/switch combinations.

    The tokenized dataset does not retain the four selected preview indices,
    so all non-fainted, non-active team members are possible switch targets.
    This intentionally gives the sanity floor the benefit of the doubt.
    """

    def __init__(self, tokenizer, dex_path, seed=0):
        self.tok = tokenizer
        self.moves = json.load(open(dex_path))["moves"]
        self.seed = seed

    def _blocks(self, ids):
        names = self.tok.decode(ids)
        return [names[self.tok.my_base + k * self.tok.mon_block:
                      self.tok.my_base + (k + 1) * self.tok.mon_block]
                for k in range(6)]

    def _slot_candidates(self, blocks, grid, slot):
        blk = next((b for b in blocks if b[0] == ("SLOT_A" if slot == 0 else "SLOT_B")), None)
        if blk is None:
            return [SlotAction("pass")], []
        damage = []
        for target, target_code in enumerate((T_FOE_A, T_FOE_B)):
            mv = int(np.argmax(grid[slot, :, target]))
            damage.append(SlotAction("move", move_slot=mv, target=target_code))
        extras = list(damage)
        for mv, token in enumerate(blk[6:10]):
            if not token.startswith("move:"):
                continue
            meta = self.moves.get(token[5:], {})
            if meta.get("category") != "Status":
                continue
            target = meta.get("target", "self")
            targets = ((T_FOE_A, T_FOE_B, T_ALLY) if target in ("normal", "any", "adjacentFoe")
                       else (T_ALLY,) if target == "adjacentAlly" else (T_AUTO,))
            extras.extend(SlotAction("move", move_slot=mv, target=t) for t in targets)
        extras.extend(SlotAction("switch", switch_to=k) for k, b in enumerate(blocks)
                      if b[0] in ("BENCH", "UNSEEN") and b[4] != "HP_0")
        return damage, list(dict.fromkeys(extras))

    def ranked(self, tokens, dmg_active, top_k=16):
        """Return deterministic ranked joint-action indices for each row."""
        result = []
        for i, (ids, grid) in enumerate(zip(tokens, dmg_active)):
            blocks = self._blocks(ids)
            da, ca = self._slot_candidates(blocks, grid, 0)
            db, cb = self._slot_candidates(blocks, grid, 1)
            head = [(a, b) for a in da for b in db if joint_ok(a, b)]
            rest = [(a, b) for a in ca for b in cb if joint_ok(a, b) and (a, b) not in head]
            random.Random(self.seed + i).shuffle(rest)
            result.append([to_index(a) * N_SLOT_ACTIONS + to_index(b)
                           for a, b in (head + rest)[:top_k]])
        return result
