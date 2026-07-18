"""The BERMUDA exercise policy: a MoveChooser with no tree search.

Per decision (the Longstaff–Schwartz exercise comparison, plan.md §2.1):

  1. screen   — every legal joint action scored by the type-chart heuristic;
                keep the top A candidates.
  2. freeze   — K scenarios: a belief-sampled opponent sheet materialized as
                a reconstructed sidecar battle, one opponent joint action
                drawn from the softmax opponent model, and the engine PRNG
                frozen via save(). Every candidate is evaluated against the
                SAME frozen scenarios (common random numbers).
  3. exercise — each (candidate, scenario) pair steps one half-turn on a
                restored fork; terminal games score ±1, live ones are
                featurized (public info only) and priced by the value net.
  4. price    — scenario values aggregate through the entropic certainty
                equivalent with state-adaptive risk λ = λ0·V̄: risk-averse
                ahead, risk-seeking behind. Argmax, or softmax at the game
                temperature.
"""

import copy
import math
import random
from pathlib import Path

import numpy as np

from actions import joint_choice, legal_slot_actions, _pos_maps
from bermuda.config import BCFG, apply_runtime_env
from bermuda.features import featurize, load_dex
from bermuda.heuristic import (active_infos, forced_choice, mon_infos,
                               request_infos, sample_joint, scored_joints)
from bermuda.model import ValueMLP
from config import CFG
from damage import DamageBridge
from data import LogParser, Side, sid
from env import Sidecar, SidecarBattle, full_set, reconstruct


def describe(joint, request, sets):
    """Human-readable 'moveA→tgt + moveB' label for logs and ChoiceInfo."""
    parts = []
    for slot, act in enumerate(joint):
        if act.kind == "pass":
            parts.append("pass")
        elif act.kind == "switch":
            name = sets[act.switch_to]["name"] if act.switch_to < len(sets) \
                else f"#{act.switch_to}"
            parts.append(f"switch→{name}")
        else:
            slot_req = (request.get("active") or [None, None])[slot]
            mv = slot_req["moves"][act.move_slot] if slot_req else {}
            tgt = {1: "→foeA", 2: "→foeB", 3: "→ally"}.get(act.target, "")
            parts.append(f"{mv.get('move', f'm{act.move_slot + 1}')}{tgt}"
                         + (" mega" if act.mega else ""))
    return " + ".join(parts)


class BermudaChooser:
    """MoveChooser implementation (agents/interfaces.py protocol)."""

    def __init__(self, ckpt, cfg=CFG, bcfg=BCFG, seed=0, debug=False,
                 n_scenarios=None, n_candidates=None, **_):
        apply_runtime_env(cfg)
        self.cfg, self.bcfg, self.debug = cfg, bcfg, debug
        self.rng = random.Random(seed)
        self.dex = load_dex(cfg)
        self.net = ValueMLP.load(Path(ckpt)) if ckpt else None
        assert self.net is not None, "BERMUDA needs a value checkpoint " \
            "(bermuda/train.py); the heuristic alone lives in paths.py"
        self.n_scenarios = n_scenarios or bcfg.n_scenarios
        self.n_candidates = n_candidates or bcfg.n_candidates
        self.sc = Sidecar(cfg)              # scenario sim, owned
        self.bridge = DamageBridge(cfg)     # required by the Bot→belief seam

    # ---- the exercise decision ------------------------------------------
    def choose(self, tracker, belief, my_id, request, brought,
               opp_brought=None, temperature=None, root_noise=None):
        opp_id = "p2" if my_id == "p1" else "p1"
        my_sets = [m.set for m in tracker.sides[my_id].mons]
        name_to_idx = {s["name"]: k for k, s in enumerate(my_sets)}
        idx_of_pos, _ = _pos_maps(request, name_to_idx)
        minfo = mon_infos(tracker, my_id, self.dex)
        foes = active_infos(tracker, opp_id, self.dex)

        pool = scored_joints(request, idx_of_pos, minfo, foes, self.dex)
        pool.sort(key=lambda js: -js[1])
        cands = [j for j, _ in pool[:self.n_candidates]]
        if len(cands) <= 1:
            joint = cands[0] if cands else sample_joint(
                request, idx_of_pos, minfo, foes, self.dex, self.rng, 0)
            return joint, self._info(joint, request, my_sets, 0.0, [], {}, {})

        scenarios, health = self._freeze_scenarios(
            tracker, belief, my_id, opp_id, my_sets, brought, opp_brought)
        if not scenarios:
            joint = cands[0]      # heuristic fallback: no scenario built
            return joint, self._info(joint, request, my_sets, 0.0, [], {},
                                     health)

        Q, opp_tally = self._exercise(cands, scenarios, my_id, opp_id,
                                      name_to_idx, health)

        # entropic certainty equivalent with state-adaptive risk (plan §2.5):
        # ahead (V̄>0) -> λ>0 risk-averse; behind (V̄<0) -> λ<0 risk-seeking
        means = np.array([np.nanmean(row) if np.any(np.isfinite(row))
                          else -1.0 for row in Q])
        lam = float(np.clip(self.bcfg.risk_lambda0 * np.nanmax(means),
                            -self.bcfg.risk_lambda0, self.bcfg.risk_lambda0))
        ces = np.array([self._ce(row, lam) for row in Q])

        if temperature and temperature > 0:
            w = np.exp((ces - ces.max()) / temperature)
            probs = w / w.sum()
            pick = int(self.rng.choices(range(len(cands)),
                                        weights=probs.tolist())[0])
        else:
            probs = (ces == ces.max()).astype(float)
            probs /= probs.sum()
            pick = int(np.argmax(ces))

        rows = sorted(zip(cands, ces, means, probs), key=lambda r: -r[1])
        table = {"rows": rows, "lam": lam}
        return cands[pick], self._info(cands[pick], request, my_sets,
                                       float(ces[pick]), rows, opp_tally,
                                       health, table)

    # ---- scenario machinery ---------------------------------------------
    def _freeze_scenarios(self, tracker, belief, my_id, opp_id, my_sets,
                          brought, opp_brought):
        health = {"recon_fail": 0, "step_fail": 0, "feat_fail": 0}
        opp_mons = tracker.sides[opp_id].mons
        if not opp_brought:
            appeared = [m.team_idx for m in opp_mons if m.appeared]
            rest = [m.team_idx for m in opp_mons if not m.appeared]
            opp_brought = (appeared + rest)[:max(1, len(brought))]
        brought_map = {my_id: list(brought), opp_id: list(opp_brought)}

        scenarios, tries = [], 0
        budget = self.n_scenarios * (1 + self.bcfg.max_recon_tries)
        while len(scenarios) < self.n_scenarios and tries < budget:
            tries += 1
            opp_team = belief.sample_sets(1, self.rng)[0]
            try:
                b, _ = reconstruct(self.sc, self.cfg.format_id, tracker,
                                   {my_id: my_sets, opp_id: opp_team},
                                   brought_map)
            except Exception:
                health["recon_fail"] += 1
                continue
            if b.ended or not b.requests.get(my_id):
                b.destroy()
                health["recon_fail"] += 1
                continue
            base = LogParser("bermuda-scn", 0, "", self.cfg.format_id)
            base.sides = {my_id: Side(my_sets),
                          opp_id: Side([full_set(s) for s in opp_team])}
            for line in b.log:
                base.feed(line)

            opp_choice, opp_desc = None, "wait"
            opp_req = b.requests.get(opp_id)
            if opp_req and not opp_req.get("wait"):
                opp_n2i = {full_set(s)["name"]: i
                           for i, s in enumerate(opp_team)}
                o_pos, _ = _pos_maps(opp_req, opp_n2i)
                o_minfo = request_infos(opp_req, o_pos,
                                        [full_set(s) for s in opp_team],
                                        self.dex)
                o_foes = active_infos(tracker, my_id, self.dex)
                o_joint = sample_joint(opp_req, o_pos, o_minfo, o_foes,
                                       self.dex, self.rng,
                                       self.bcfg.opp_temp,
                                       self.bcfg.opp_uniform_mix)
                opp_choice = joint_choice(opp_req, o_joint, opp_n2i)
                opp_desc = describe(o_joint, opp_req,
                                    [full_set(s) for s in opp_team])
            scenarios.append({"state": b.save(), "opp_choice": opp_choice,
                              "opp_desc": opp_desc, "base": base})
            b.destroy()
        return scenarios, health

    def _exercise(self, cands, scenarios, my_id, opp_id, name_to_idx,
                  health):
        """Q[a, k] over CRN-paired one-half-turn forks; batched net at end."""
        Q = np.full((len(cands), len(scenarios)), np.nan, dtype=np.float64)
        feats, turns, where = [], [], []
        opp_tally = {}
        for ki, scn in enumerate(scenarios):
            opp_tally[scn["opp_desc"]] = opp_tally.get(scn["opp_desc"], 0) + 1
            for ai, joint in enumerate(cands):
                fork = SidecarBattle.restore(self.sc, scn["state"])
                try:
                    my_req = fork.requests.get(my_id)
                    choices = {my_id: joint_choice(my_req, joint,
                                                   name_to_idx)}
                    if scn["opp_choice"] and fork.requests.get(opp_id) \
                            and not fork.requests[opp_id].get("wait"):
                        choices[opp_id] = scn["opp_choice"]
                    resp = fork.step(choices)
                    logs = list(resp["log"])
                    if resp["errors"]:
                        health["step_fail"] += 1
                        resp = fork.step(
                            {s: "default" for s in resp["errors"]})
                        logs += resp["log"]
                    if fork.ended:
                        Q[ai, ki] = (0.0 if fork.winner not in ("p1", "p2")
                                     else (1.0 if fork.winner == my_id
                                           else -1.0))
                    else:
                        t2 = copy.deepcopy(scn["base"])
                        for line in logs:
                            t2.feed(line)
                        feats.append(featurize(t2, my_id, self.dex))
                        turns.append(min(t2.turn_no, 25))
                        where.append((ai, ki))
                except Exception:
                    health["feat_fail"] += 1
                finally:
                    try:
                        fork.destroy()
                    except Exception:
                        pass
        if feats:
            vals = self.net.predict_np(np.stack(feats),
                                       np.array(turns, dtype=np.int64))
            for (ai, ki), v in zip(where, vals):
                Q[ai, ki] = float(v)
        return Q, opp_tally

    @staticmethod
    def _ce(row, lam):
        """Entropic certainty equivalent of one candidate's scenario values."""
        vals = row[np.isfinite(row)]
        if len(vals) == 0:
            return -1.0
        if abs(lam) < 1e-3:
            return float(vals.mean())
        z = -lam * vals
        m = z.max()
        return float(-(m + math.log(np.exp(z - m).mean())) / lam)

    # ---- protocol extras -------------------------------------------------
    def forced_choice(self, tracker, my_id, request):
        """Matchup-aware forced-switch line (agent_server routes here)."""
        opp_id = "p2" if my_id == "p1" else "p1"
        name_to_idx = {m.set["name"]: m.team_idx
                       for m in tracker.sides[my_id].mons}
        idx_of_pos, _ = _pos_maps(request, name_to_idx)
        return forced_choice(request, idx_of_pos,
                             mon_infos(tracker, my_id, self.dex),
                             active_infos(tracker, opp_id, self.dex),
                             self.dex, self.rng)

    def _info(self, joint, request, my_sets, value, rows, opp_tally, health,
              table=None):
        n = max(1, sum(opp_tally.values()))
        return {
            "value": value, "solve": False,
            "strategy": [(describe(j, request, my_sets), float(p))
                         for j, _, _, p in rows[:6]] or
                        [(describe(joint, request, my_sets), 1.0)],
            "q": [[describe(j, request, my_sets), float(ce), float(mn)]
                  for j, ce, mn, _ in rows[:8]],
            "opp_pred": sorted(((d, c / n) for d, c in opp_tally.items()),
                               key=lambda dp: -dp[1]),
            "health": {**health, "lam": (table or {}).get("lam", 0.0)},
        }

    def close(self):
        self.sc.close()
        self.bridge.close()
