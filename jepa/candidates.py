"""Candidate joint actions synthesized from stored window tensors.

The v3 BC scoring heads (``score_head``/``opp_score_head``) are trained like
v2's candidate CE: rank the human's joint action against sampled legal
alternatives, every candidate scored through a full action-conditioned pass of
the dynamics ``T``. The seq shards don't store candidate sets, but they don't
need to — everything required to rebuild a side's action menu is already in
the position tensors (move ids in ``mon_cat``, active/fainted/bench/mega flags
in ``mon_scalar``), so negatives are built here on the fly, with no re-prep.

One :class:`CandidateBuilder` serves both the trainer (sampled negatives per
shard, disk-cached) and ``probe_strategy.py`` (full-menu enumeration for
joint-ranking metrics). The menu construction mirrors
``jepa.features.legal_my_joints`` semantics: each active mon's known move
slots x that move's real target codes (from the dex), plus switches to live
bench mons, plus mega variants when the mon's mega is available — a superset
of true legality (PP/trapping unknown), exactly what the play-time ranking
sees. The *other* side's active tokens are marked ``AK_UNK`` ("acted,
unobserved"), so a candidate's score is the consequence of one side's move
with the opponent marginalized — v2's consequence scoring re-expressed
through v3's dynamics.
"""

import random

import numpy as np

from actions import (SlotAction, T_ALLY, T_AUTO, T_FOE_A, T_FOE_B, joint_ok,
                     to_index)
from jepa.features import (AK_MOVE, AK_SWITCH, AK_UNK, MC_MV0, MON_TOKENS,
                           MS_ACTIVE, MS_FAINTED, MS_MEGA_AVAIL, MS_SLOT_A,
                           MS_SLOT_B, N_ACT_FIELDS, N_MON, N_MOVES)

# move-id target classes -> legal target codes (mirrors legal_my_joints)
_TGT_AUTO, _TGT_SINGLE, _TGT_ALLY = 0, 1, 2
_TARGETS = {_TGT_AUTO: (T_AUTO,), _TGT_SINGLE: (T_FOE_A, T_FOE_B, T_ALLY),
            _TGT_ALLY: (T_ALLY,)}


class CandidateBuilder:
    """Rebuild a side's joint-action menu + action arrays from window tensors."""

    def __init__(self, vocab, n_neg=7, seed=0):
        """Precompute id-indexed move tables (bp/priority/target class)."""
        self.n_neg = n_neg
        self.rng = random.Random(seed)
        n = len(vocab.moves)
        self.bp = np.zeros(n, dtype=np.int64)
        self.prio = np.zeros(n, dtype=np.int64)
        self.tgt = np.full(n, _TGT_SINGLE, dtype=np.int64)   # UNK -> 3 targets
        for name, i in vocab.moves.items():
            if name in ("__pad__", "__unk__"):
                continue
            _, bp, prio, _ = vocab.move_meta(name)
            self.bp[i] = int(min(bp, 250))
            self.prio[i] = int(prio)
            t = vocab.move_target(name)
            self.tgt[i] = (_TGT_SINGLE if t in ("normal", "any", "adjacentFoe")
                           else _TGT_ALLY if t == "adjacentAlly" else _TGT_AUTO)

    # -- menu reconstruction --------------------------------------------------
    def menu(self, mcat, mscal, side="my"):
        """Enumerate one side's joint actions from a stored ``[12, ...]`` row.

        Returns ``(joints, active, moves)`` where ``joints`` is a list of
        ``(SlotAction, SlotAction)``, ``active`` maps slot -> team index and
        ``moves[k]`` is that mon's stored move-id list."""
        base = 0 if side == "my" else N_MON
        active, bench, moves = {}, [], {}
        for j in range(N_MON):
            k = base + j
            moves[j] = [int(mcat[k, MC_MV0 + i]) for i in range(N_MOVES)]
            if mscal[k, MS_FAINTED] > 0.5:
                continue
            if mscal[k, MS_SLOT_A] > 0.5:
                active[0] = j
            elif mscal[k, MS_SLOT_B] > 0.5:
                active[1] = j
            else:
                bench.append(j)

        def slot_acts(slot):
            """Menu for one slot: switches + each move x its target codes."""
            k = active.get(slot)
            if k is None:
                return [SlotAction("pass")]
            acts = [SlotAction("switch", switch_to=j) for j in bench]
            mega = mscal[base + k, MS_MEGA_AVAIL] > 0.5
            for i, mid in enumerate(moves[k]):
                if mid == 0:
                    continue
                for t in _TARGETS[int(self.tgt[mid])]:
                    acts.append(SlotAction("move", move_slot=i, target=t))
                    if mega:
                        acts.append(SlotAction("move", move_slot=i, target=t,
                                               mega=True))
            return acts or [SlotAction("pass")]

        s0, s1 = slot_acts(0), slot_acts(1)
        joints = [(a, b) for a in s0 for b in s1 if joint_ok(a, b)]
        return joints, active, moves

    # -- action arrays --------------------------------------------------------
    def arrays(self, joints, active, moves, mscal, side="my"):
        """``[C, 12, 7]`` action arrays; the other side's actives are AK_UNK."""
        base = 0 if side == "my" else N_MON
        other = N_MON - base
        out = np.zeros((len(joints), MON_TOKENS, N_ACT_FIELDS), dtype=np.int16)
        for j in range(N_MON):
            if mscal[other + j, MS_ACTIVE] > 0.5:
                out[:, other + j, 0] = AK_UNK
        for c, joint in enumerate(joints):
            for slot in (0, 1):
                k = active.get(slot)
                if k is None:
                    continue
                a = joint[slot]
                row = out[c, base + k]
                if a.kind == "move":
                    mid = moves[k][a.move_slot]
                    row[0], row[1], row[2] = AK_MOVE, mid, a.target
                    row[3] = int(a.mega)
                    row[5], row[6] = self.bp[mid], self.prio[mid] + 6
                elif a.kind == "switch":
                    row[0], row[4] = AK_SWITCH, a.switch_to + 1
        return out

    # -- trainer negatives ----------------------------------------------------
    def negatives(self, mcat, mscal, true_slot, mask, side="my"):
        """Sampled negative candidates for every masked row of a shard.

        ``mcat``/``mscal`` are the step-0 position arrays ``[N, 12, ...]``,
        ``true_slot`` the human per-slot indices ``[N, 2]``. Returns
        ``(negs [N, n_neg, 12, 7] int16, neg_mask [N, n_neg] bool)``; the
        human's own joint is excluded from the pool so candidate 0 (the stored
        positive) is the unique correct answer."""
        n = mcat.shape[0]
        negs = np.zeros((n, self.n_neg, MON_TOKENS, N_ACT_FIELDS),
                        dtype=np.int16)
        neg_mask = np.zeros((n, self.n_neg), dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            joints, active, moves = self.menu(mcat[i], mscal[i], side)
            human = (int(true_slot[i][0]), int(true_slot[i][1]))
            pool = [j for j in joints
                    if (to_index(j[0]), to_index(j[1])) != human]
            if not pool:
                continue
            self.rng.shuffle(pool)
            pick = pool[:self.n_neg]
            negs[i, :len(pick)] = self.arrays(pick, active, moves, mscal[i],
                                              side)
            neg_mask[i, :len(pick)] = True
        return negs, neg_mask
