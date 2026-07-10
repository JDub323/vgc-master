"""Decoupled-UCT node.

Turns are simultaneous, so a node keeps TWO independent bandit tables — one
per player — instead of the single table of alternating-move UCT. Each player
selects its own action by PUCT over its own statistics; the joint (i, j) pair
indexes the child. Neither table conditions on the other player's pick, which
is what makes the visit distribution converge toward a mixed strategy at
equilibrium points instead of an exploitable pure strategy.

Values are stored from the searching player's perspective throughout; the
opponent's table just accumulates the negation (zero-sum).
"""

import numpy as np


class Node:
    """One action-history point in one determinized game. The sim state at a
    node varies visit-to-visit (the sidecar RNG rolls damage/accuracy fresh
    each descent), so stats average over chance implicitly — no chance nodes."""

    def __init__(self, my_actions, opp_actions, my_priors, opp_priors, value=0.0):
        """Store action lists and initialize parallel float64 P/N/W arrays."""
        self.my_actions = my_actions        # list of joint (SlotAction, SlotAction)
        self.opp_actions = opp_actions
        self.my_p = np.asarray(my_priors, dtype=np.float64)
        self.opp_p = np.asarray(opp_priors, dtype=np.float64)
        self.my_n = np.zeros(len(my_actions))
        self.my_w = np.zeros(len(my_actions))
        self.opp_n = np.zeros(len(opp_actions))
        self.opp_w = np.zeros(len(opp_actions))
        self.n = 0
        self.value = value                  # net value at expansion (diagnostics)
        self.children = {}                  # (i, j) -> Node

    @staticmethod
    def _pick(p, n, w, total, c_puct):
        """Return the integer argmax of zero-FPU PUCT for one player."""
        q = np.where(n > 0, w / np.maximum(n, 1), 0.0)   # unvisited: neutral FPU
        return int(np.argmax(q + c_puct * p * np.sqrt(total + 1) / (1 + n)))

    def select(self, c_puct):
        """Each player independently PUCT-maximizes over its own table."""
        i = self._pick(self.my_p, self.my_n, self.my_w, self.n, c_puct)
        j = self._pick(self.opp_p, self.opp_n, self.opp_w, self.n, c_puct)
        return i, j

    def update(self, i, j, z):
        """z in [-1, 1] from the searching player's perspective."""
        self.n += 1
        self.my_n[i] += 1
        self.my_w[i] += z
        self.opp_n[j] += 1
        self.opp_w[j] -= z
