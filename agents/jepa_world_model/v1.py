"""JEPA world-model chooser: latent one-ply planning + matrix-game solve.

Implements ``agents.interfaces.MoveChooser``. At each move request it encodes
the live position, enumerates candidate joint actions for both players,
predicts the payoff of every pairing with the learned world model (averaged
over belief determinizations), solves the resulting matrix game for a mixed
strategy, and samples it at the play temperature. See ``JEPA_DESIGN.md``.
"""

import random
import sys

import numpy as np
import torch

from actions import (T_AUTO, T_FOE_A, T_FOE_B, SlotAction, from_index, joint_ok,
                     legal_joint_actions, to_index)
from actions import _pos_maps
from config import CFG
from jepa.config import JCFG
from jepa.features import FeatureExtractor, action_arrays
from jepa.solver import solve_matrix

N_SLOT_ACTIONS = 39


class JEPAWorldModelChooser:
    """Plan one joint action with a latent world model and a matrix solve."""

    def __init__(self, model, vocab, cfg=CFG, jcfg=JCFG, seed=0, bridge=None):
        """Bind a loaded model + vocab; take an optional shared damage bridge."""
        self.model = model.eval()
        for p in self.model.parameters():        # inference-only chooser
            p.requires_grad_(False)
        self.vocab = vocab
        self.cfg = cfg
        self.jcfg = jcfg
        self.extractor = FeatureExtractor(vocab)
        self.rng = random.Random(seed)
        self.bridge = bridge
        self.device = next(model.parameters()).device

    # -- MoveChooser contract ------------------------------------------------
    def choose(self, tracker, belief, my_id, request, brought,
               opp_brought=None, temperature=None, root_noise=None):
        """Return one legal ``JointAction`` plus display/training ``ChoiceInfo``."""
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
            return self._plan(view, legal, belief, brought, opp_brought, temp)
        except Exception as exc:                      # never forfeit on a net bug
            print(f"[jepa] planning fell back: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return legal[0], _info("fallback:first-legal")

    def close(self):
        """Close the owned damage bridge, if any."""
        if self.bridge is not None:
            self.bridge.close()

    # -- planning ------------------------------------------------------------
    def _plan(self, view, legal, belief, brought, opp_brought, temp):
        """Encode, enumerate candidates, build V, solve, and sample."""
        summary = belief.summary() if belief is not None else {}
        dmg = self._damage(view, belief)
        # single belief-summary pass drives candidate selection
        pos0 = self.extractor.extract(view, summary, brought=brought,
                                      opp_brought=opp_brought, dmg=dmg)
        z0 = self.model.encode(self._batch([pos0]))
        my_logits, opp_logits = self.model.policies(z0)
        my_logits = my_logits[0].cpu().numpy()
        opp_logits = opp_logits[0].cpu().numpy()

        my_cands = _top_joints(legal, my_logits, self.jcfg.top_k_mine)
        opp_cands = _top_joints(self._opp_actions(view, belief), opp_logits,
                                self.jcfg.top_k_opp)
        na, nb = len(my_cands), len(opp_cands)

        # determinizations: sampled opponent movesets marginalize hidden sets
        dets = (belief.sample_sets(self.jcfg.n_determinizations, self.rng)
                if belief is not None else [None])
        v = np.zeros((na, nb), dtype=np.float64)
        for det in dets:
            opp_ms = _movesets(det)
            pos = self.extractor.extract(view, summary, brought=brought,
                                         opp_brought=opp_brought,
                                         opp_movesets=opp_ms, dmg=dmg)
            v += self._payoffs(pos, my_cands, opp_cands)
        v /= len(dets)

        p_row, p_col, val = solve_matrix(v, self.jcfg.solver_iters)
        idx = self._sample(p_row, temp)
        chosen = my_cands[idx]
        q = v @ p_col
        info = {
            "value": float(val), "solve": False,
            "strategy": _decoded(my_cands, p_row),
            "q": _decoded(my_cands, q),
            "opp_pred": _decoded(opp_cands, p_col),
            "health": {"candidates": float(na), "opp_candidates": float(nb),
                       "determinizations": float(len(dets))},
        }
        return chosen, info

    def _payoffs(self, pos, my_cands, opp_cands):
        """World-model value for every (mine, opp) candidate pair -> ``[na,nb]``."""
        acts, na, nb = [], len(my_cands), len(opp_cands)
        for a in my_cands:
            for b in opp_cands:
                acts.append(action_arrays(pos, a, b, self.vocab))
        act = torch.as_tensor(np.stack(acts), dtype=torch.long, device=self.device)
        batch = self._batch([pos])
        z = self.model.encode(batch).expand(na * nb, -1, -1)
        dmg = batch["dmg"].expand(na * nb, -1, -1)
        with torch.no_grad():
            zp = self.model.predict(z, act, dmg)
            val = self.model.value(zp).cpu().numpy()
        return val.reshape(na, nb)

    def _damage(self, view, belief):
        """Compute the ally->foe damage matrix if a bridge and belief exist."""
        if self.bridge is None or belief is None:
            return None
        from damage import damage_features
        try:
            return damage_features(view, belief, self.bridge)
        except Exception:                              # bridge hiccup -> no edges
            return None

    def _opp_actions(self, view, belief):
        """Construct a plausible opponent joint-action list from beliefs."""
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

    # -- tensor plumbing -----------------------------------------------------
    def _batch(self, positions):
        """Stack a list of :class:`Position` into a device tensor dict."""
        t = lambda arr, dt: torch.as_tensor(np.stack(arr), dtype=dt,
                                            device=self.device)
        return {
            "gcat": t([p.global_cat for p in positions], torch.long),
            "gscal": t([p.global_scalar for p in positions], torch.float32),
            "mcat": t([p.mon_cat for p in positions], torch.long),
            "mscal": t([p.mon_scalar for p in positions], torch.float32),
            "dmg": t([p.dmg_edge for p in positions], torch.float32),
        }

    def _sample(self, p_row, temp):
        """Pick a candidate index: argmax at ``temp==0``, else tempered sample."""
        if temp <= 1e-6:
            return int(np.argmax(p_row))
        w = np.power(np.clip(p_row, 1e-9, None), 1.0 / temp)
        w = w / w.sum()
        return int(self.rng.choices(range(len(w)), weights=w)[0])


# ---- helpers ---------------------------------------------------------------
def _movesets(team):
    """Sampled team -> 6 x up-to-4 move sids in preview order (``None`` -> [])."""
    if team is None:
        return None
    return [[mv for mv in s.get("moves", [])][:4] for s in team]


def _opp_slot_actions(team, k, moveset):
    """Believed legal slot actions for the opponent mon ``k`` (``None`` -> pass)."""
    if k is None:
        return [SlotAction("pass")]
    acts = [SlotAction("switch", switch_to=o["team_idx"]) for o in team
            if o["team_idx"] != k and not o["fainted"] and o["active_slot"] is None]
    for j, _mv in enumerate(moveset.get(k, [])[:4]):
        for tgt in (T_FOE_A, T_FOE_B, T_AUTO):
            acts.append(SlotAction("move", move_slot=j, target=tgt))
    return acts or [SlotAction("pass")]


def _joint_score(joint, logits):
    """Sum of per-slot log-softmax scores for one joint slot-action pair."""
    return logits[0][to_index(joint[0])] + logits[1][to_index(joint[1])]


def _top_joints(joints, logits, k):
    """Return the ``k`` highest-scoring joint actions under per-slot ``logits``."""
    lsm = logits - logits.max(-1, keepdims=True)
    lsm = lsm - np.log(np.exp(lsm).sum(-1, keepdims=True))
    ranked = sorted(joints, key=lambda j: _joint_score(j, lsm), reverse=True)
    return ranked[:max(1, k)]


def _decoded(joints, weights):
    """Pair each joint action's readable description with its weight, sorted."""
    rows = sorted(zip(joints, weights), key=lambda r: -r[1])
    return [(_desc(j), float(w)) for j, w in rows if w > 1e-4][:6]


def _desc(joint):
    """Render a joint action as a short human-readable string."""
    return ", ".join(_slot_desc(a) for a in joint)


def _slot_desc(a):
    """Render one slot action (pass/move+target/switch) as a string."""
    if a.kind == "pass":
        return "pass"
    if a.kind == "switch":
        return f"switch->{a.switch_to}"
    tgt = {T_AUTO: "", T_FOE_A: " foeA", T_FOE_B: " foeB", 3: " ally"}[a.target]
    return f"move{a.move_slot}{tgt}{' mega' if a.mega else ''}"


def _info(desc, value=0.0):
    """Minimal ``ChoiceInfo`` for degenerate/forced decisions."""
    return {"value": value, "solve": False, "strategy": [(desc, 1.0)],
            "q": [], "opp_pred": [], "health": {}}


def build_jepa_chooser(ckpt=None, cfg=CFG, jcfg=JCFG, seed=0):
    """Construct a chooser from a checkpoint, or a random-init net if absent.

    The vocabulary is restored from the checkpoint when present, else rebuilt
    from ``dex.json`` + ``vocab.json``. A damage bridge is created when
    available (it needs the Node calc install) and silently skipped otherwise,
    so the agent runs on a bare box with only degraded damage edges.
    """
    from pathlib import Path

    from beliefs import load_dex
    from jepa.vocab import JEPAVocab
    from models.jepa_wm import JEPAWorldModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(ckpt) if ckpt else None
    model = None
    if ckpt and ckpt.exists():
        try:
            model, vstate = JEPAWorldModel.load(ckpt, device)
            vocab = (JEPAVocab.from_state(vstate, load_dex(cfg)) if vstate
                     else JEPAVocab.build(cfg))
        except Exception as exc:      # not a JEPA checkpoint (e.g. the DUCT default)
            print(f"[jepa] '{ckpt}' is not a JEPA checkpoint ({exc}); "
                  "using a random-init net", file=sys.stderr)
            model = None
    if model is None:
        vocab = JEPAVocab.build(cfg)
        model = JEPAWorldModel(vocab.sizes(), jcfg, vocab.state()).to(device)
    bridge = None
    if jcfg.use_damage_features:
        try:
            from damage import DamageBridge
            bridge = DamageBridge(cfg)
        except Exception as exc:                       # no Node install -> skip
            print(f"[jepa] damage bridge unavailable ({exc}); using no edges",
                  file=sys.stderr)
    return JEPAWorldModelChooser(model, vocab, cfg, jcfg, seed, bridge)
