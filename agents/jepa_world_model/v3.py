"""v3 Strategy-JEPA chooser: latent matrix-tree search over recursive dynamics.

Depth-1 (default, ladder-fast): encode the position, take top-k own x top-k
believed-opponent joint candidates, apply the latent dynamics ``T(Z, a, b)``
once per pairing, read the value head off each predicted latent, average the
payoff matrix over belief determinizations, and regret-match it into a mixed
strategy (v1's game solve at v2's speed).

Depth-2 (``JEPAConfig.plan_depth = 2``; generation-time improvement operator):
instead of reading the value at depth-1 leaves, each leaf recurses — child
candidates come from the policy heads on the predicted latents, the active-mon
maps are propagated through any switch actions, and the leaf value is the
solved value of the child matrix. Entirely sim-free. Approximation, documented:
faint-forced switches inside the lookahead are not modeled (the value head
reads latents that do encode faints, so the leaf value still reflects them).
"""

import sys

import numpy as np
import torch

from actions import (SlotAction, T_AUTO, T_FOE_A, T_FOE_B, _pos_maps,
                     joint_ok, legal_joint_actions)
from agents.jepa_world_model.v1 import (_decoded, _info, _joint_score,
                                        _movesets, _opp_slot_actions,
                                        _top_joints)
from config import CFG
from jepa.config import JCFG
from jepa.features import FeatureExtractor, action_arrays
from jepa.solver import solve_matrix_anchored

N_MOVES = 4


class JEPAStrategyChooser:
    """Choose a joint action by solving latent payoff matrices over ``T``."""

    def __init__(self, model, vocab, cfg=CFG, jcfg=JCFG, seed=0, bridge=None):
        """Bind a loaded strategy model + vocab; optional damage bridge."""
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.vocab = vocab
        self.cfg = cfg
        self.jcfg = jcfg
        self.extractor = FeatureExtractor(vocab)
        import random
        self.rng = random.Random(seed)
        self.bridge = bridge
        self.device = next(model.parameters()).device
        self.record = False           # self-play hook, same contract as v2
        self.last_plan = None

    # -- MoveChooser contract ------------------------------------------------
    def choose(self, tracker, belief, my_id, request, brought,
               opp_brought=None, temperature=None, root_noise=None):
        """Return one legal ``JointAction`` plus display ``ChoiceInfo``."""
        temp = self.jcfg.play_temperature if temperature is None else temperature
        view = tracker._view(my_id)
        name_to_idx = {m.set["name"]: m.team_idx
                       for m in tracker.sides[my_id].mons}
        idx_of_pos = _pos_maps(request, name_to_idx)[0]
        legal = legal_joint_actions(request, idx_of_pos)
        if not legal:
            return (SlotAction("pass"), SlotAction("pass")), _info("no legal move")
        if len(legal) == 1:
            return legal[0], _info("forced", 0.0)
        try:
            return self._plan(view, legal, belief, temp)
        except Exception as exc:
            print(f"[jepa-s] planning fell back: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return legal[0], _info("fallback:first-legal")

    def close(self):
        """Close the owned damage bridge, if any."""
        if self.bridge is not None:
            self.bridge.close()

    # -- planning ------------------------------------------------------------
    def _plan(self, view, legal, belief, temp):
        """Root matrix over candidates; leaves via T (+ child solve at depth 2)."""
        summary = belief.summary() if belief is not None else {}
        dmg = self._damage(view, belief)
        dets = (belief.sample_sets(self.jcfg.n_determinizations, self.rng)
                if belief is not None else [None])
        v_sum = None
        my_cands = opp_cands = None
        my_prior = opp_prior = None
        first_pos = None
        for det in dets:
            pos = self.extractor.extract(view, summary,
                                         opp_movesets=_movesets(det), dmg=dmg)
            z = self.model.encode(self._batch(pos, 1))
            if my_cands is None:
                first_pos = pos
                opp_joints = self._opp_actions(view, belief)
                if getattr(self.jcfg, "prior_from_score", True):
                    # joint action-conditioned BC prior: every candidate is
                    # scored through T with the other side marginalized
                    # (AK_UNK) — v2's consequence scoring. At eta=0 the solve
                    # returns this prior, i.e. BC play over the full legal set.
                    my_cands, my_prior = self._score_rank(
                        z, pos, legal, mine=True, k=self.jcfg.top_k_mine_s)
                    opp_cands, opp_prior = self._score_rank(
                        z, pos, opp_joints, mine=False,
                        k=self.jcfg.top_k_opp_s)
                else:                     # factorized per-slot prior (legacy)
                    my_logits, opp_logits = self.model.policies(z)
                    my_lg = my_logits[0].cpu().numpy()
                    opp_lg = opp_logits[0].cpu().numpy()
                    my_cands = _top_joints(legal, my_lg,
                                           self.jcfg.top_k_mine_s)
                    opp_cands = _top_joints(opp_joints, opp_lg,
                                            self.jcfg.top_k_opp_s)
                    my_prior = _joint_prior(my_cands, my_lg)
                    opp_prior = _joint_prior(opp_cands, opp_lg)
            v = self._matrix(z, pos, my_cands, opp_cands,
                             depth=self.jcfg.plan_depth)
            v_sum = v if v_sum is None else v_sum + v
        v = v_sum / len(dets)

        p_row, p_col, val = solve_matrix_anchored(
            v, my_prior, opp_prior, self.jcfg.solver_eta,
            self.jcfg.solver_iters)
        idx = self._sample(p_row, temp)
        if self.record:
            self.last_plan = {"pos": first_pos, "cands": my_cands,
                              "chosen": idx, "p_row": p_row, "matrix": v}
        info = {"value": float(val), "solve": False,
                "strategy": _decoded(my_cands, p_row),
                "q": _decoded(my_cands, v @ p_col),
                "opp_pred": _decoded(opp_cands, p_col),
                "health": {"candidates": float(len(my_cands)),
                           "opp_candidates": float(len(opp_cands)),
                           "determinizations": float(len(dets)),
                           "depth": float(self.jcfg.plan_depth),
                           "score_prior": float(getattr(
                               self.jcfg, "prior_from_score", True))}}
        return my_cands[idx], info

    def _score_rank(self, z, pos, joints, mine, k):
        """Rank a side's joint candidates by its BC score head through ``T``.

        Returns the top-``k`` joints and a softmax prior over them."""
        n = len(joints)
        if mine:
            acts = np.stack([action_arrays(pos, j, None, self.vocab)
                             for j in joints])
        else:
            acts = np.stack([action_arrays(pos, None, j, self.vocab)
                             for j in joints])
        act_t = torch.as_tensor(acts, dtype=torch.long, device=self.device)
        dmg_t = torch.as_tensor(pos.dmg_edge, dtype=torch.float32,
                                device=self.device)[None].expand(n, -1, -1)
        zp = self.model.step(z.expand(n, -1, -1), act_t, dmg_t)
        s = (self.model.score(zp) if mine
             else self.model.opp_score(zp)).cpu().numpy()
        idx = np.argsort(-s)[:max(1, k)]
        e = np.exp(s[idx] - s[idx].max())
        return [joints[i] for i in idx], e / e.sum()

    def _matrix(self, z, pos, my_cands, opp_cands, depth):
        """Payoff matrix ``[na, nb]`` via one batched ``T`` application/pair."""
        na, nb = len(my_cands), len(opp_cands)
        acts = np.stack([action_arrays(pos, a, b, self.vocab)
                         for a in my_cands for b in opp_cands])
        act_t = torch.as_tensor(acts, dtype=torch.long, device=self.device)
        z_rep = z.expand(na * nb, -1, -1)
        dmg_t = torch.as_tensor(pos.dmg_edge, dtype=torch.float32,
                                device=self.device)[None].expand(na * nb, -1, -1)
        zp = self.model.step(z_rep, act_t, dmg_t)
        if depth <= 1:
            return self.model.value(zp).cpu().numpy().reshape(na, nb)
        vals = np.empty(na * nb)
        for i, (a, b) in enumerate((a, b) for a in my_cands for b in opp_cands):
            vals[i] = self._child_value(zp[i:i + 1], pos, a, b, depth - 1)
        return vals.reshape(na, nb)

    def _child_value(self, zp, pos, a, b, depth):
        """Leaf value at depth >= 2: solve the anchored child matrix under ``zp``."""
        my_act, opp_act = _propagate_active(pos, a, b)
        my_c, my_p = self._latent_cands(zp, my_act, pos.my_movesets, mine=True)
        opp_c, opp_p = self._latent_cands(zp, opp_act, pos.opp_movesets,
                                          mine=False)
        if not my_c or not opp_c:
            return float(self.model.value(zp).cpu().numpy()[0])
        import dataclasses
        child_pos = dataclasses.replace(pos, my_active=my_act,
                                        opp_active=opp_act)
        na, nb = len(my_c), len(opp_c)
        acts = np.stack([action_arrays(child_pos, ca, cb, self.vocab)
                         for ca in my_c for cb in opp_c])
        act_t = torch.as_tensor(acts, dtype=torch.long, device=self.device)
        zc = self.model.step(zp.expand(na * nb, -1, -1), act_t, None)
        v = self.model.value(zc).cpu().numpy().reshape(na, nb)
        return float(solve_matrix_anchored(v, my_p, opp_p,
                                           self.jcfg.solver_eta,
                                           self.jcfg.solver_iters)[2])

    def _latent_cands(self, zp, active, movesets, mine):
        """Child joint candidates + prior, ranked by the policy heads on ``zp``."""
        my_logits, opp_logits = self.model.policies(zp)
        logits = (my_logits if mine else opp_logits)[0].cpu().numpy()
        slot_acts = [_slot_actions_from_map(active, s, movesets)
                     for s in (0, 1)]
        joints = [(x, y) for x in slot_acts[0] for y in slot_acts[1]
                  if joint_ok(x, y)]
        cands = _top_joints(joints, logits, self.jcfg.child_k)
        return cands, _joint_prior(cands, logits)

    def _opp_actions(self, view, belief):
        """Believed opponent joint actions (same construction as v1)."""
        team = view["opp"]["team"]
        active = {m["active_slot"]: m["team_idx"] for m in team
                  if m["active_slot"] is not None and not m["fainted"]}
        moveset = {}
        for k in active.values():
            if belief is not None:
                moveset[k] = [mv for mv in belief.top_particle(k)["moves"]][:4]
            else:
                moveset[k] = [mv for mv in team[k]["revealed_moves"]][:4]
        slot = [_opp_slot_actions(team, active.get(s), moveset) for s in (0, 1)]
        return [(a, b) for a in slot[0] for b in slot[1] if joint_ok(a, b)]

    def _damage(self, view, belief):
        """Ally->foe damage edges when a bridge and belief exist."""
        if self.bridge is None or belief is None:
            return None
        from damage import damage_features
        try:
            return damage_features(view, belief, self.bridge)
        except Exception:
            return None

    def _batch(self, pos, n):
        """Replicate one position into an ``n``-row device tensor dict."""
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device
                                          ).unsqueeze(0).expand(n, *a.shape)
        return {"gcat": t(pos.global_cat, torch.long),
                "gscal": t(pos.global_scalar, torch.float32),
                "mcat": t(pos.mon_cat, torch.long),
                "mscal": t(pos.mon_scalar, torch.float32),
                "dmg": t(pos.dmg_edge, torch.float32)}

    def _sample(self, p_row, temp):
        """Argmax at ``temp==0``, else a tempered sample of the mixed strategy."""
        if temp <= 1e-6:
            return int(np.argmax(p_row))
        w = np.power(np.clip(p_row, 1e-9, None), 1.0 / temp)
        w = w / w.sum()
        return int(self.rng.choices(range(len(w)), weights=w)[0])


def _joint_prior(joints, logits):
    """Softmax prior over candidate joints from per-slot policy logits."""
    lsm = logits - logits.max(-1, keepdims=True)
    lsm = lsm - np.log(np.exp(lsm).sum(-1, keepdims=True))
    scores = np.array([_joint_score(j, lsm) for j in joints])
    e = np.exp(scores - scores.max())
    return e / e.sum()


# ---- depth >= 2 helpers -----------------------------------------------------
def _propagate_active(pos, a, b):
    """Advance the active-mon maps through any switch actions in ``(a, b)``.

    Moves/passes leave the maps unchanged; faint-forced replacements are not
    modeled (documented stage-1 approximation)."""
    my_act = dict(pos.my_active)
    opp_act = dict(pos.opp_active)
    for slot, act in enumerate(a):
        if act.kind == "switch":
            my_act[slot] = act.switch_to
    for slot, act in enumerate(b):
        if act.kind == "switch":
            opp_act[slot] = act.switch_to
    return my_act, opp_act


def _slot_actions_from_map(active, slot, movesets):
    """Plausible slot actions for a propagated active map (child expansion)."""
    k = active.get(slot)
    if k is None:
        return [SlotAction("pass")]
    on_field = set(active.values())
    acts = [SlotAction("switch", switch_to=j) for j in range(6)
            if j not in on_field and movesets[j]]
    for j in range(min(N_MOVES, len(movesets[k]))):
        for tgt in (T_FOE_A, T_FOE_B, T_AUTO):
            acts.append(SlotAction("move", move_slot=j, target=tgt))
    return acts or [SlotAction("pass")]


def build_jepa_strategy_chooser(ckpt=None, cfg=CFG, jcfg=JCFG, seed=0):
    """Build the v3 chooser from a checkpoint, or random-init when absent."""
    from pathlib import Path

    from beliefs import load_dex
    from jepa.vocab import JEPAVocab
    from models.jepa_strategy import JEPAStrategyModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(ckpt) if ckpt else None
    model = None
    if ckpt and ckpt.exists():
        try:
            model, vstate = JEPAStrategyModel.load(ckpt, device)
            vocab = (JEPAVocab.from_state(vstate, load_dex(cfg)) if vstate
                     else JEPAVocab.build(cfg))
        except Exception as exc:
            print(f"[jepa-s] '{ckpt}' is not a strategy checkpoint ({exc}); "
                  "using a random-init net", file=sys.stderr)
            model = None
    if model is None:
        vocab = JEPAVocab.build(cfg)
        model = JEPAStrategyModel(vocab.sizes(), jcfg, vocab.state()).to(device)
    mjcfg = model.jcfg
    # play-time overrides for the eta ladder / prior ablations without
    # re-exporting: VGC_JEPA_ETA=0 is the BC floor (v2-inside-v3),
    # VGC_JEPA_SCORE_PRIOR=0 falls back to the factorized per-slot prior
    import dataclasses
    import os
    eta = os.environ.get("VGC_JEPA_ETA")
    sp = os.environ.get("VGC_JEPA_SCORE_PRIOR")
    if eta is not None or sp is not None:
        mjcfg = dataclasses.replace(
            mjcfg,
            solver_eta=float(eta) if eta is not None else mjcfg.solver_eta,
            prior_from_score=(sp not in ("0", "false")) if sp is not None
            else getattr(mjcfg, "prior_from_score", True))
        print(f"[jepa-s] overrides: eta={mjcfg.solver_eta} "
              f"score_prior={mjcfg.prior_from_score}", file=sys.stderr)
    bridge = None
    if getattr(mjcfg, "use_damage_features", False):
        try:
            from damage import DamageBridge
            bridge = DamageBridge(cfg)
        except Exception as exc:
            print(f"[jepa-s] damage bridge unavailable ({exc}); no edges",
                  file=sys.stderr)
    return JEPAStrategyChooser(model, vocab, cfg, mjcfg, seed, bridge)
