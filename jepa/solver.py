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


def solve_matrix_anchored(m, prior_row, prior_col, eta=4.0, iters=200):
    """Prior-anchored logit equilibrium of a zero-sum matrix game.

    Mirror-descent fixed point ``p ∝ prior_row · exp(eta · M q)`` /
    ``q ∝ prior_col · exp(−eta · Mᵀ p)``, iterates averaged. At ``eta → 0``
    this returns the priors (pure policy play); ``eta → ∞`` approaches the
    Nash of ``M`` (pure value play). In between, the value head only moves
    the decision where its payoff differences exceed its noise floor — the
    stable blend of a strong BC prior with a young value head. Returns
    ``(row_strategy, col_strategy, value)``.
    """
    m = np.asarray(m, dtype=np.float64)
    nr, nc = m.shape
    p = np.asarray(prior_row, dtype=np.float64).clip(1e-9)
    q = np.asarray(prior_col, dtype=np.float64).clip(1e-9)
    p, q = p / p.sum(), q / q.sum()
    lp0, lq0 = np.log(p), np.log(q)
    sum_p, sum_q = np.zeros(nr), np.zeros(nc)
    for _ in range(iters):
        p = _softmax_log(lp0 + eta * (m @ q))
        q = _softmax_log(lq0 - eta * (m.T @ p))
        sum_p += p
        sum_q += q
    avg_p = sum_p / sum_p.sum()
    avg_q = sum_q / sum_q.sum()
    return avg_p, avg_q, float(avg_p @ m @ avg_q)


def _softmax_log(logits):
    """Numerically stable softmax of a 1-D log-space vector."""
    e = np.exp(logits - logits.max())
    return e / e.sum()
