"""Lead and switch-in selection with the FROZEN baseline value head.

The experiment's premise is that the base model already knows more about
positions than the trivial team-preview/forced-switch policies use. These
selectors extract that knowledge without touching a single weight: build the
hypothetical public state each candidate decision would produce, tokenize it
exactly the way training did (same tokenizer, same belief summary, same
damage features), and read the value head.

Team preview is a simultaneous one-shot game, so a single argmax against one
assumed opponent lead would be exploitable. Instead the selector forms a
payoff matrix over (my expert-pruned bring/lead combos) x (their expert-ranked
lead pairs) and scores each of my combos as
``alpha * weighted-mean + (1 - alpha) * min`` — an opponent-model/maximin
blend. The value head cannot see WHICH four I bring (turn-1 states carry no
bring information, only leads), so the back pair falls to the expert coverage
ranking inside the pruned combo list.

Forced switches are evaluated directly: each legal replacement's post-switch
state (averaged over the opponent's own replacement scenarios when they are
also forced to switch) goes through the value head; argmax wins. No sim
stepping is needed — switching changes only who stands in the slot.
"""

import copy

from actions import _pos_maps
from agents.lead_switch.expert import (ExpertLeadSelector, preview_order)
from agents.lead_switch.lscfg import LSCFG
from damage import damage_features


def _clear_actives(team_view):
    """Reset every mon's active slot in one side's view; return None."""
    for m in team_view:
        m["active_slot"] = None


def _put_active(team_view, idx, slot):
    """Place preview index ``idx`` into active ``slot`` in a view; return None."""
    m = team_view[idx]
    m["active_slot"] = slot
    m["appeared"] = True
    m["turns_active"] = m.get("turns_active") or 0


class _HypoEval:
    """Batched value-head evaluation of hypothetical CTS view dicts."""

    def __init__(self, model, tok, bridge=None):
        """Keep shared (never owned) model, tokenizer, and damage bridge."""
        self.model, self.tok, self.bridge = model, tok, bridge

    def values(self, states, belief):
        """Return one searching-side value per state (single net batch).

        The damage matrix depends only on sets/boosts/field — none of which
        differ between hypotheticals that merely move mons in and out of
        slots — so it is computed once from the first state and shared."""
        import numpy as np
        summary = belief.summary()
        dmg = damage_features(states[0], belief, self.bridge) \
            if self.bridge else {}
        toks = np.stack([self.tok.encode(s, summary, dmg) for s in states])
        _, values, _ = self.model.predict_batch(toks)
        return [float(v) for v in values]


class ValueLeadSelector:
    """Frozen-net payoff-matrix team preview chooser (expert-pruned)."""

    def __init__(self, model, tok, cfg, bridge=None, ls=LSCFG):
        """Wire the shared net/tokenizer/bridge and an expert pruner."""
        self.ls = ls
        self.eval = _HypoEval(model, tok, bridge)
        self.expert = ExpertLeadSelector(cfg, bridge, ls)

    def choose(self, tracker, belief, my_id, n_bring=None):
        """Return ``(order, info)`` like the expert selector."""
        ls = self.ls
        my_sets = [m.set for m in tracker.sides[my_id].mons]
        rows = self.expert.score_combos(my_sets, belief, n_bring)[:ls.v_my_top]
        off, dfn, _ = self.expert.matchup.tables(my_sets, belief)
        opp_ws = self.expert.opp_lead_weights(
            len(belief.species), off, dfn)
        opp_ws.sort(key=lambda tw: -tw[1])
        opp_ws = opp_ws[:ls.v_opp_top]
        tot = sum(w for _, w in opp_ws)
        opp_ws = [(T, w / tot) for T, w in opp_ws]

        base = tracker._view(my_id)
        base["turn"] = 1
        states, index = [], []
        for c, (_, lead, back) in enumerate(rows):
            for q, (T, _) in enumerate(opp_ws):
                s = copy.deepcopy(base)
                _clear_actives(s["my"]["team"])
                _clear_actives(s["opp"]["team"])
                for slot, i in enumerate(lead):
                    _put_active(s["my"]["team"], i, slot)
                for slot, k in enumerate(T):
                    _put_active(s["opp"]["team"], k, slot)
                states.append(s)
                index.append((c, q))
        vals = self.eval.values(states, belief)
        payoff = [[0.0] * len(opp_ws) for _ in rows]
        for (c, q), v in zip(index, vals):
            payoff[c][q] = v

        best_c, best_score = 0, float("-inf")
        for c in range(len(rows)):
            mean = sum(w * payoff[c][q] for q, (_, w) in enumerate(opp_ws))
            worst = min(payoff[c])
            score = ls.v_maximin_alpha * mean + (1 - ls.v_maximin_alpha) * worst
            if score > best_score:
                best_c, best_score = c, score
        _, lead, back = rows[best_c]
        info = {"kind": "value", "score": best_score,
                "expert_rank": best_c,
                "opp_pairs": [list(T) for T, _ in opp_ws],
                "payoff_row": payoff[best_c]}
        return preview_order(lead, back), info


class ValueSwitchSelector:
    """Frozen-net forced-switch chooser over hypothetical post-switch states."""

    def __init__(self, model, tok, cfg, bridge=None, ls=LSCFG):
        """Wire the shared net/tokenizer/bridge."""
        self.ls = ls
        self.eval = _HypoEval(model, tok, bridge)

    def _opp_scenarios(self, tracker, opp_id):
        """Opponent replacement scenarios for their empty/fainted slots.

        Under CTS their brought four is inferred the same way the chooser
        does it: mons that appeared, filled to four by preview order."""
        opp = tracker.sides[opp_id]
        empty = [s for s in (0, 1)
                 if (m := opp.active(s)) is None or m.fainted]
        bench_alive = [m.team_idx for m in opp.mons
                       if m.active_slot is None and not m.fainted]
        brought = [m.team_idx for m in opp.mons if m.appeared]
        brought += [m.team_idx for m in opp.mons
                    if not m.appeared][:max(0, 4 - len(brought))]
        cands = [k for k in bench_alive if k in brought]
        if not empty or not cands:
            return [{}]
        from itertools import permutations
        outs = [dict(zip(empty, pick))
                for pick in permutations(cands, min(len(empty), len(cands)))]
        return outs[:self.ls.v_opp_switch_cap] or [{}]

    def choose(self, request, tracker, belief, my_id):
        """Return one Showdown choice string for a forceSwitch request."""
        from itertools import permutations
        opp_id = "p2" if my_id == "p1" else "p1"
        me = tracker.sides[my_id]
        name_to_idx = {m.set["name"]: k for k, m in enumerate(me.mons)}
        idx_of_pos, pos_of_idx = _pos_maps(request, name_to_idx)

        forced_slots = [s for s, f in
                        enumerate(request.get("forceSwitch") or []) if f]
        cands = []
        for pos, p in enumerate(request["side"]["pokemon"], start=1):
            if p["active"] or p["condition"] == "0 fnt":
                continue
            try:
                cands.append((pos, idx_of_pos(pos)))
            except KeyError:
                continue
        if not forced_slots or not cands:
            return ", ".join("pass" for _ in
                             (request.get("forceSwitch") or ["x"]))

        assigns = list(permutations(cands, min(len(forced_slots), len(cands))))
        scenarios = self._opp_scenarios(tracker, opp_id)
        base = tracker._view(my_id)
        states, index = [], []
        for a, assign in enumerate(assigns):
            for sc in scenarios:
                s = copy.deepcopy(base)
                for slot, (_, idx) in zip(forced_slots, assign):
                    for m in s["my"]["team"]:
                        if m["active_slot"] == slot:
                            m["active_slot"] = None
                    _put_active(s["my"]["team"], idx, slot)
                for slot, k in sc.items():
                    for m in s["opp"]["team"]:
                        if m["active_slot"] == slot:
                            m["active_slot"] = None
                    _put_active(s["opp"]["team"], k, slot)
                states.append(s)
                index.append(a)
        vals = self.eval.values(states, belief)
        totals = [0.0] * len(assigns)
        counts = [0] * len(assigns)
        for a, v in zip(index, vals):
            totals[a] += v
            counts[a] += 1
        best = max(range(len(assigns)),
                   key=lambda a: (totals[a] / max(1, counts[a]), -a))
        chosen = dict(zip(forced_slots, assigns[best]))
        out = []
        for slot, f in enumerate(request.get("forceSwitch") or []):
            if f and slot in chosen:
                out.append(f"switch {chosen[slot][0]}")
            else:
                out.append("pass")
        return ", ".join(out)
