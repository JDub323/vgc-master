"""Benchmark floors. Both expose predict_batch(dmg_active) -> per-slot action
distributions [B, 2, N_SLOT_ACTIONS] so evaluate.py scores them exactly like
the network. dmg_active is the prepped [B, my_slot, move, opp_slot] avg-damage%
grid (uint8)."""

import numpy as np

from actions import N_SLOT_ACTIONS, SlotAction, T_FOE_A, T_FOE_B, to_index


class RandomPolicy:
    def predict_batch(self, dmg_active):
        b = len(dmg_active)
        return np.full((b, 2, N_SLOT_ACTIONS), 1.0 / N_SLOT_ACTIONS)


class MaxDamagePolicy:
    """Picks, per slot, the (move, target) with the highest expected immediate
    damage. Slots with no damage information fall back to uniform."""

    def __init__(self, eps=0.05):
        self.eps = eps   # smoothing so log-loss is finite

    def predict_batch(self, dmg_active):
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
