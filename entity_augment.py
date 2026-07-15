"""Permutation augmentation for entity-hybrid training (train_entity.py --augment).

The point: layout 3 fixes "mon 3" and "move slot 2" as physical positions, so
a flat model can memorize conventions (Protect is usually the fourth move)
instead of semantics. Reordering teams and move slots — with every dependent
label remapped — teaches permutation-equivariance directly.

Per batch row, independently:
  - opponent team order: permute the six opponent mon blocks, their belief
    blocks, and the aux oracle labels (items/abilities/moves) identically.
    No action labels reference opponent team order (targets are slot-coded),
    so nothing else moves.
  - opponent move slots: permute each opponent mon's four revealed-move
    tokens. The aux move target is multi-hot (order-free), so label-safe.
  - my team order: permute my six mon blocks AND remap switch action labels
    (switch action 33+k targets team-preview slot k, actions.py).
  - my move slots: permute each of my mons' four move tokens AND remap move
    action labels (move action encodes move_slot) for whichever of my mons is
    active in slot A / slot B, identified by its SLOT_A / SLOT_B token.

The damage block (tokens [273:561]) is deliberately NOT kept consistent: its
axes are (my mon i, my move j, opp mon k) and would need a triple remap, but
EntityHybridNet drops those tokens entirely. Never feed augmented tokens to a
model that reads the damage block.

All permutations are sampled uniformly (identity included), vectorized with
numpy take_along_axis — no Python loop over the batch.
"""

import numpy as np

from tokenizer import MON_BLOCK, N_MONS

MON_T = MON_BLOCK + 1     # 18
MY_BASE = 15
OPP_BASE = MY_BASE + N_MONS * MON_T
BELIEF_BASE = OPP_BASE + N_MONS * MON_T
BEL_T = 7
DMG_BASE = BELIEF_BASE + N_MONS * BEL_T
MOVE_LO, MOVE_HI = 6, 10
N_MOVE_ACTS = 32          # slot actions 1..32 are moves; 33..38 switches


def _rand_perms(rng, B, n, k):
    """[B, n, k] independent uniform permutations of range(k) (n=1 -> [B, k])."""
    perm = np.argsort(rng.random((B, n, k)), axis=-1)
    return perm[:, 0] if n == 1 else perm


def _permute_blocks(view, perm):
    """Reorder [B, 6, T] blocks so new[b, j] = old[b, perm[b, j]]."""
    return np.take_along_axis(view, perm[:, :, None], axis=1)


def augment_batch(tokens, acts, opp_items, opp_abils, opp_moves, rng,
                  slot_a_id, slot_b_id):
    """Return augmented copies of (tokens, acts, opp_items, opp_abils,
    opp_moves). Inputs are one batch's numpy arrays; slot ids come from the
    tokenizer vocab (``vocab["SLOT_A"]``, ``vocab["SLOT_B"]``)."""
    B = tokens.shape[0]
    tokens = tokens.copy()
    acts = acts.copy()
    my = tokens[:, MY_BASE:OPP_BASE].reshape(B, N_MONS, MON_T)
    opp = tokens[:, OPP_BASE:BELIEF_BASE].reshape(B, N_MONS, MON_T)
    bel = tokens[:, BELIEF_BASE:DMG_BASE].reshape(B, N_MONS, BEL_T)

    # active-mon lookup BEFORE any reordering: which of my team-preview mons
    # holds slot A / slot B right now (its move actions carry move_slot)
    slot_tok = my[:, :, 0]
    a_mask, b_mask = slot_tok == slot_a_id, slot_tok == slot_b_id
    active = np.stack([a_mask.argmax(1), b_mask.argmax(1)], 1)      # [B, 2]
    has_active = np.stack([a_mask.any(1), b_mask.any(1)], 1)        # [B, 2]

    # -- my move slots: permute tokens, remap move action labels -------------
    r = _rand_perms(rng, B, N_MONS, 4)                              # [B, 6, 4]
    my[:, :, MOVE_LO:MOVE_HI] = np.take_along_axis(
        my[:, :, MOVE_LO:MOVE_HI], r, axis=2)
    inv_r = np.argsort(r, axis=2)          # old move slot -> new move slot
    rows = np.arange(B)
    for s in (0, 1):
        a = acts[:, s]
        move = (a >= 1) & (a <= N_MOVE_ACTS) & has_active[:, s]
        ms, rest = (a - 1) // 8, (a - 1) % 8
        new_ms = inv_r[rows, active[:, s], np.clip(ms, 0, 3)]
        acts[:, s] = np.where(move, 1 + new_ms * 8 + rest, a)

    # -- my team order: permute blocks, remap switch action labels -----------
    q = _rand_perms(rng, B, 1, N_MONS)                              # [B, 6]
    my[:] = _permute_blocks(my, q)
    inv_q = np.argsort(q, axis=1)          # old team idx -> new team idx
    for s in (0, 1):
        a = acts[:, s]
        sw = a > N_MOVE_ACTS
        new_idx = inv_q[rows, np.clip(a - N_MOVE_ACTS - 1, 0, N_MONS - 1)]
        acts[:, s] = np.where(sw, 1 + N_MOVE_ACTS + new_idx, a)

    # -- opponent team order: blocks + beliefs + aux labels together ---------
    p = _rand_perms(rng, B, 1, N_MONS)
    opp[:] = _permute_blocks(opp, p)
    bel[:] = _permute_blocks(bel, p)
    opp_items = np.take_along_axis(opp_items, p, axis=1)
    opp_abils = np.take_along_axis(opp_abils, p, axis=1)
    opp_moves = np.take_along_axis(opp_moves, p[:, :, None], axis=1)

    # -- opponent move slots: tokens only (aux move loss is multi-hot) -------
    r_opp = _rand_perms(rng, B, N_MONS, 4)
    opp[:, :, MOVE_LO:MOVE_HI] = np.take_along_axis(
        opp[:, :, MOVE_LO:MOVE_HI], r_opp, axis=2)

    return tokens, acts, opp_items, opp_abils, opp_moves
