"""Simultaneous-move MCTS: decoupled UCT with policy priors, determinized
over belief-sampled opponent sets, running on reconstructed sidecar battles.

Per move decision:
  1. Sample K opponent teams from the particle filter (`n_determinizations`).
  2. For each sample, rebuild the current public state as a fresh sidecar
     battle with those sets as ground truth (env.reconstruct) — hidden info
     is fixed per determinization, which turns each search into a
     perfect-information game.
  3. Run sims_per_move / K simulations per determinization. Each simulation
     restores a fork of the root, descends the tree applying both players'
     PUCT picks as real sim steps (RNG fresh each visit, so chance is
     averaged implicitly), expands one leaf, evaluates it with the value head
     (from the searching player's view), and backs the value up both bandit
     tables. Priors come from the policy net run from BOTH perspectives;
     actions are pruned to the net's top-k joints (`top_k_actions`).
  4. In endgames (<= solve_endgame_at mons per side) the value head is
     ignored: no pruning, and every simulation runs to actual game end, so
     leaves are exact wins/losses. This is what makes the Metagross/Kingambit
     scenario an honest equilibrium check.
  5. Aggregate root visit counts across determinizations: that distribution
     IS the output mixed strategy. Play samples from it with temperature.

Known v1 approximations (all documented in the README): reconstruction restores
the public Protect streak but drops other volatiles (choice lock, encore,
PP spent), forced switch-ins inside simulations are random, and within a
determinization the simulated opponent 'sees' our true sets (standard
determinization paranoia).
"""

import copy
import random
import time
from collections import Counter

import numpy as np

# _pos_maps and joint_choice live in actions.py (the leaf module) and are
# re-exported here for existing importers of search.mcts.
from actions import (T_ALLY, T_AUTO, T_FOE_A, T_FOE_B, _pos_maps,
                     joint_choice, joint_index, legal_joint_actions)
from agents.encoding.v1 import TokenPositionEncoder
from agents.evaluators.v1 import PolicyValueLeafEvaluator
from agents.priors.v1 import PolicyValuePrior
from agents.search.v1 import DecoupledUCTSearcher
from beliefs import OpponentBelief, determinized
from config import CFG
from damage import DamageBridge
from data import LogParser, Side, sid
from env import Sidecar, full_set, reconstruct
from search.debug import SearchDebug, belief_report, root_table
from search.node import Node

TGT = {T_AUTO: "", T_FOE_A: ">1", T_FOE_B: ">2", T_ALLY: ">ally"}


def _joint_priors(joint_dist, joints, k):
    """Model joint dist (flat [N_JOINT_ACTIONS], either head architecture —
    see policy_value.predict_batch) -> normalized priors over legal joints,
    optionally pruned to the top-k (k=None in solve mode: exactness beats
    pruning when the branching is already small)."""
    return PolicyValuePrior().legal_priors(joint_dist, joints, k)


class DetGame:
    """One determinization: a reconstructed battle whose opponent sets are a
    single sample from the belief filter, plus everything needed to step it
    and tokenize its leaves."""

    def __init__(self, searcher, tracker, opp_sample, my_id, my_request,
                 my_brought, opp_brought, solve):
        """Reconstruct one sampled hidden team and create its evaluated root."""
        cfg = searcher.cfg
        self.my = my_id
        self.opp = "p2" if my_id == "p1" else "p1"
        self.solve = solve
        my_sets = [m.set for m in tracker.sides[self.my].mons]
        opp_sets = [full_set(s) for s in opp_sample]
        teams = {self.my: my_sets, self.opp: opp_sets}
        brought = {self.my: my_brought, self.opp: opp_brought}
        self.name_to_idx = {
            self.my: {s["name"]: k for k, s in enumerate(my_sets)},
            self.opp: {s["name"]: k for k, s in enumerate(opp_sets)}}

        # tracker seeded with the determinized "truth", so leaf tokenization
        # sees the sampled sets as the opponent's own team
        seed = LogParser("det", 0, "", cfg.format_id)
        seed.sides = {self.my: copy.deepcopy(tracker.sides[self.my]),
                      self.opp: Side(opp_sets)}
        real = tracker.sides[self.opp]
        for m, rm in zip(seed.sides[self.opp].mons, real.mons):
            m.species_cur, m.hp, m.status = rm.species_cur, rm.hp, rm.status
            m.boosts = dict(rm.boosts)
            m.fainted, m.active_slot = rm.fainted, rm.active_slot
            m.appeared, m.mega_done = rm.appeared, rm.mega_done
            m.transformed, m.turns_active = rm.transformed, rm.turns_active
            m.protect_ct = rm.protect_ct        # public: everyone saw it protect
            m.revealed_moves = list(rm.revealed_moves)
            m.revealed_item, m.item_consumed = rm.revealed_item, rm.item_consumed
            m.revealed_ability = rm.revealed_ability
        seed.sides[self.opp].mega_used = real.mega_used
        seed.sides[self.opp].conditions = dict(real.conditions)
        seed.weather, seed.terrain = tracker.weather, tracker.terrain
        seed.trickroom, seed.turn_no = tracker.trickroom, tracker.turn_no
        self.seed_tracker = seed

        # collapsed beliefs: inside this determinization both sides' sets are
        # fixed, and the simulated opponent sees our true team (the usual
        # determinization simplification)
        self.bel_opp = determinized(opp_sets, cfg)
        self.bel_me = determinized(my_sets, cfg)
        self.sum_opp = self.bel_opp.summary()
        self.sum_my = self.bel_me.summary()

        b, self.orders = reconstruct(searcher.sc, cfg.format_id, tracker,
                                     teams, brought)
        assert not b.ended
        # root: OUR actions come from the real request (true legality —
        # disabled moves, trapping); the opponent's from the reconstruction
        self.root = searcher._expand(self, b, seed, my_request=my_request)
        self.root_state = b.save()
        b.destroy()


class DeterminizedDUCTChooser:
    """Determinized decoupled-UCT chooser composed from versioned bricks."""

    def __init__(self, model, tok, cfg=CFG, seed=0, debug=False, sidecar=None,
                 position_encoder=None, policy_prior=None,
                 leaf_evaluator=None, searcher=None):
        """Inject or create the v1 bricks and owned simulator/calc resources."""
        self.model, self.tok, self.cfg = model, tok, cfg
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        # A caller can hand us a Sidecar to share (self-play/gate workers pass
        # the game sidecar, so one node process serves the whole worker instead
        # of one-per-searcher). Battles are keyed by id and a worker's calls are
        # sequential, so sharing is safe. Only close what we own.
        self._own_sc = sidecar is None
        self.sc = Sidecar(cfg) if sidecar is None else sidecar
        self.bridge = DamageBridge(cfg) if cfg.use_damage_features else None
        self.position_encoder = position_encoder or \
            TokenPositionEncoder(tok, self.bridge)
        self.policy_prior = policy_prior or PolicyValuePrior()
        self.leaf_evaluator = leaf_evaluator or PolicyValueLeafEvaluator(model)
        self.searcher = searcher or DecoupledUCTSearcher()
        # The battle/game layer owns this object's lifecycle. Exposing the
        # expected class lets AgentSpec-loaded games construct the recorded
        # belief implementation without moving updates inside the chooser.
        self.belief_model_cls = OpponentBelief
        self.dbg = SearchDebug(debug)
        self.health = Counter()

    def close(self):
        """Shut down this searcher's node processes (benchmark/self-play
        workers spawn many searchers; long-lived scripts should not leak).
        An injected sidecar is owned by the caller and left open."""
        if self._own_sc:
            self.sc.close()
        if self.bridge:
            self.bridge.close()

    # -- one move decision ---------------------------------------------------
    def choose(self, tracker, belief, my_id, my_request, my_brought,
               opp_brought=None, temperature=None, policy_only=False,
               root_noise=None):
        """Returns (joint SlotAction pair sampled from the root mixed
        strategy, info dict with the full strategy / value / opponent
        prediction for display). policy_only=True skips the simulations and
        plays straight from the net's root priors — the no-search bot for
        the Elo gap and for fast human games. root_noise=(eps, alpha) mixes
        Dirichlet noise into OUR root priors (self-play exploration, the
        AlphaZero recipe) — never set during evaluation or live play."""
        cfg = self.cfg
        t_start = time.perf_counter()
        self.health = Counter()
        opp_id = "p2" if my_id == "p1" else "p1"
        opp_mons = tracker.sides[opp_id].mons
        if opp_brought is None:   # CTS: we only know what appeared; fill by preview order
            n = min(4, len(opp_mons))
            opp_brought = [m.team_idx for m in opp_mons if m.appeared]
            opp_brought += [m.team_idx for m in opp_mons
                            if not m.appeared][:max(0, n - len(opp_brought))]
        my_alive = sum(not tracker.sides[my_id].mons[k].fainted for k in my_brought)
        opp_alive = sum(not opp_mons[k].fainted for k in opp_brought)
        solve = (my_alive <= cfg.solve_endgame_at
                 and opp_alive <= cfg.solve_endgame_at) and not policy_only
        assert self.model is not None or solve, \
            "without a trained model only endgame (solve-to-terminal) search works"

        with self.dbg("det_build"):
            dets = [DetGame(self, tracker, sample, my_id, my_request,
                            my_brought, opp_brought, solve)
                    for sample in belief.sample_sets(
                        1 if policy_only else cfg.n_determinizations, self.rng)]
        if root_noise and not policy_only:
            eps, alpha = root_noise
            for det in dets:
                noise = self.np_rng.dirichlet([alpha] * len(det.root.my_p))
                det.root.my_p = (1 - eps) * det.root.my_p + eps * noise
        if not policy_only:
            budget = max(1, cfg.sims_per_move // len(dets))
            self.searcher.run(self, dets, budget)

        # the aggregated root visit distribution IS the mixed strategy
        # (policy-only: the root priors stand in for visits)
        strat = self.searcher.aggregate_root(dets, policy_only)
        visits = np.array([s[1] for s in strat])
        t = cfg.play_temperature if temperature is None else temperature
        if t <= 0:
            pick = 0
        else:
            weights = visits ** (1.0 / t)
            pick = self.rng.choices(range(len(strat)), weights=weights)[0]

        probs = visits / visits.sum()
        opp_pred = {}
        for det in dets:
            for a, p in zip(det.root.opp_actions, det.root.opp_p):
                d = self._describe(det.seed_tracker, det.opp, a)
                opp_pred[d] = opp_pred.get(d, 0.0) + p / len(dets)
        if policy_only:
            value = float(np.mean([d.root.value for d in dets]))
        else:
            value = float(sum(d.root.my_w.sum() for d in dets)
                          / max(1, sum(d.root.n for d in dets)))
        wall = time.perf_counter() - t_start
        info = {
            "value": value,
            "solve": solve,
            # machine-readable mixed strategy over flat joint indices —
            # the self-play policy target (visit counts, unnormalized)
            "visits": [(joint_index(s[0][0], s[0][1]), float(s[1]))
                       for s in strat],
            "strategy": [(self._describe(tracker, my_id, s[0]), float(p))
                         for s, p in zip(strat, probs)],
            "q": [(self._describe(tracker, my_id, s[0]),
                   float(s[2] / max(1, s[1]))) for s in strat],
            "opp_pred": sorted(opp_pred.items(), key=lambda kv: -kv[1])[:5],
            "health": dict(self.health) | {"wall_s": wall},
        }
        if self.dbg.enabled:
            self._debug_print(dets, belief, wall, policy_only)
        return strat[pick][0], info

    def _debug_print(self, dets, belief, wall, policy_only):
        """Print phase, health, root, and posterior diagnostics; return ``None``."""
        h = self.health
        sims = int(h["sims"])
        print(f"\n[search debug] {len(dets)} det(s), {sims} sims, {wall:.2f}s"
              + (f" ({sims / wall:.0f} sims/s)" if sims else " (policy only)"))
        print(self.dbg.report(wall))
        steps = max(1, int(h["steps"]))
        print(f"health: invalid-action fallbacks {int(h['fallbacks'])} "
              f"({h['fallbacks'] / steps:.1%} of steps — reconstruction "
              f"fidelity signal), forced switches {int(h['forced_switches'])}, "
              f"terminals {int(h['terminals'])}, value leaves "
              f"{int(h['value_leaves'])}, depth avg "
              f"{h['depth_sum'] / max(1, sims):.1f} max {int(h['depth_max'])}, "
              f"leaf-eval depth avg "
              f"{h['leaf_depth_sum'] / max(1, int(h['value_leaves'])):.2f} "
              f"(rollout_depth={self.cfg.rollout_depth})")
        if self.bridge:
            tot = max(1, self.bridge.hits + self.bridge.misses)
            print(f"bridge cache: {self.bridge.hits}/{tot} hits "
                  f"({self.bridge.hits / tot:.0%}), "
                  f"{len(self.bridge.cache)} entries")
        print(root_table(
            dets, lambda det, a: self._describe(det.seed_tracker, det.my, a)))
        print(belief_report(belief))

    # -- internals -------------------------------------------------------
    def _leaf_value(self, det, b, trk, leaf):
        """Value backed up from a freshly expanded leaf. With rollout_depth=1
        this is just the net value at the leaf (AlphaZero default). With depth D
        we play D-1 extra plies of the real sim, both sides taking the greedy
        (argmax-prior) joint action, then evaluate the net at the deeper
        position — so a leaf that only *looks* good because it preserves HP
        (double-Protect) is scored after the opponent's follow-up actually
        happens. A game that ends inside the lookahead returns the true result."""
        return self.searcher.leaf_value(self, det, b, trk, leaf)

    def _settle(self, b, trk):
        """Play out forced switches inside a simulation (random legal switch,
        a documented v1 simplification) until both sides face move requests."""
        return self.searcher.settle(self, b, trk)

    def _expand(self, det, b, trk, my_request=None):
        """Return a new ``Node`` from legal actions, encoded views, and priors."""
        self.health["expands"] += 1
        my_req = my_request or b.requests[det.my]
        opp_req = b.requests[det.opp]
        my_joints = legal_joint_actions(
            my_req, _pos_maps(my_req, det.name_to_idx[det.my])[0])
        opp_joints = legal_joint_actions(
            opp_req, _pos_maps(opp_req, det.name_to_idx[det.opp])[0])
        if self.model is None:
            return Node(my_joints, opp_joints,
                        np.full(len(my_joints), 1.0 / len(my_joints)),
                        np.full(len(opp_joints), 1.0 / len(opp_joints)))

        if (hasattr(self.position_encoder, "position") and
                hasattr(self.position_encoder, "encode_position")):
            with self.dbg("views"):
                pos_my = self.position_encoder.position(
                    trk, det.my, det.bel_opp)
                pos_opp = self.position_encoder.position(
                    trk, det.opp, det.bel_me)
            with self.dbg("encode"):
                toks = np.stack([
                    self.position_encoder.encode_position(pos_my, det.sum_opp),
                    self.position_encoder.encode_position(pos_opp, det.sum_my),
                ])
        else:
            with self.dbg("encode"):
                toks = np.stack([
                    self.position_encoder.encode(
                        trk, det.my, det.bel_opp, det.sum_opp),
                    self.position_encoder.encode(
                        trk, det.opp, det.bel_me, det.sum_my),
                ])
        with self.dbg("net"):
            dists, values, _ = self.leaf_evaluator.predict_batch(toks)
        k = None if det.solve else self.cfg.top_k_actions
        my_p, my_joints = self.policy_prior.legal_priors(
            dists[0], my_joints, k)
        opp_p, opp_joints = self.policy_prior.legal_priors(
            dists[1], opp_joints, k)
        return Node(my_joints, opp_joints, my_p, opp_p,
                    self.leaf_evaluator.value(values, 0))

    def _describe(self, tracker, side_id, joint):
        """Human-readable joint action, e.g. 'suckerpunch>1, sw garchomp'."""
        side = tracker.sides[side_id]
        out = []
        for slot, a in enumerate(joint):
            m = side.active(slot)
            if a.kind == "pass" or (a.kind == "move" and m is None):
                out.append("pass")
            elif a.kind == "switch":
                out.append("sw " + sid(side.mons[a.switch_to].species_cur))
            else:
                mv = m.set["moves"][a.move_slot] \
                    if a.move_slot < len(m.set["moves"]) else f"m{a.move_slot + 1}"
                out.append(mv + TGT[a.target] + ("+mega" if a.mega else ""))
        return ", ".join(out)
