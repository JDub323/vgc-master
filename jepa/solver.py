"""Two-player zero-sum matrix-game solver (regret matching).

The planner produces a payoff matrix ``M[a, b]`` = the searching side's
predicted win probability when it plays candidate ``a`` and the opponent plays
candidate ``b``. VGC win probability is zero-sum, so the row player maximizes
and the column player minimizes. Regret-matching+ averages to a Nash mixed
strategy — which is what makes the agent play mixed at matching-pennies nodes
(e.g. Sucker Punch dilemmas) instead of a pure, exploitable action.
"""

import numpy as np


def solve_matrix(m, iters=256):
    """Return ``(row_strategy, col_strategy, value)`` for payoff matrix ``m``.

    ``m`` is ``[n_row, n_col]`` (row payoff). Uses regret-matching+ with
    averaged strategies; ``value`` is the averaged saddle value ``pᵀ m q``.
    """
    m = np.asarray(m, dtype=np.float64)
    nr, nc = m.shape
    if nr == 0 or nc == 0:
        return np.ones(max(nr, 1)) / max(nr, 1), np.ones(max(nc, 1)) / max(nc, 1), 0.0
    if nr == 1 and nc == 1:
        return np.array([1.0]), np.array([1.0]), float(m[0, 0])
    reg_r = np.zeros(nr)
    reg_c = np.zeros(nc)
    sum_r = np.zeros(nr)
    sum_c = np.zeros(nc)
    p = np.ones(nr) / nr
    q = np.ones(nc) / nc
    for _ in range(iters):
        p = _match(reg_r)
        q = _match(reg_c)
        sum_r += p
        sum_c += q
        # row player's payoff of each pure action against current q (maximize)
        u_r = m @ q
        reg_r = np.maximum(reg_r + (u_r - p @ u_r), 0.0)     # RM+
        # column player minimizes row payoff, i.e. maximizes -mᵀp
        u_c = -(m.T @ p)
        reg_c = np.maximum(reg_c + (u_c - q @ u_c), 0.0)
    avg_r = sum_r / sum_r.sum()
    avg_c = sum_c / sum_c.sum()
    return avg_r, avg_c, float(avg_r @ m @ avg_c)


def _match(regret):
    """Regret-matching strategy: normalize positive regrets, else uniform."""
    pos = np.maximum(regret, 0.0)
    s = pos.sum()
    return pos / s if s > 0 else np.ones_like(regret) / len(regret)
