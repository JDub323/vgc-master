"""Expert-system lead selection and forced-switch selection.

Two hard-coded selectors built on the MatchupModel damage tables:

ExpertLeadSelector — team preview as a one-shot simultaneous game played
approximately: enumerate all C(6,2)*C(4,2) = 90 bring/lead combinations,
score each lead pair against a softmax-weighted distribution over the
opponent's 15 possible lead pairs (they pick good leads too), add a coverage
term for the four brought (no opponent mon should wall the whole bring), and
take the argmax. Small documented synergy bonuses (Fake Out, redirection,
own speed control under a slow pair) stand in for the doubles lead lore that
pure pairwise damage misses.

ExpertSwitchSelector — the Game Freak trainer-AI recipe adapted to doubles
and beliefs: for every legal replacement, expected best-move damage OUT to
the live opponent actives minus danger-weighted damage IN from their likely
sets (scaled up when the replacement is chipped), plus a speed tiebreak.
Deterministic argmax, greedy across multiple forced slots.

Neither selector touches the frozen DUCT chooser; they only answer the two
request kinds it never sees.
"""

from itertools import combinations

from actions import _pos_maps
from agents.lead_switch.lscfg import LSCFG
from agents.lead_switch.matchup import (FAKEOUT_MOVES, MatchupModel,
                                        REDIRECT_MOVES, SPEED_CONTROL_MOVES)


def enumerate_previews(n_team, n_bring):
    """Yield every ``(lead_pair, back_tuple)`` bring choice, preview indices.

    lead_pair is an (i, j) with i < j; back_tuple the remaining brought
    indices sorted. 90 combos for the standard 6-mon / bring-4 case."""
    for bring in combinations(range(n_team), min(n_bring, n_team)):
        for lead in combinations(bring, min(2, len(bring))):
            back = tuple(k for k in bring if k not in lead)
            yield lead, back


def preview_order(lead, back):
    """Flatten one combo into the order list ``brought`` and choice digits."""
    return list(lead) + list(back)


def team_choice_string(order):
    """Preview-index order -> Showdown ``team`` choice (1-based digits)."""
    return "team " + "".join(str(i + 1) for i in order)


class ExpertLeadSelector:
    """Hard-coded bring-4/lead-2 chooser over MatchupModel tables."""

    def __init__(self, cfg, bridge=None, ls=LSCFG):
        """Build the matchup model; ``bridge`` may be shared with the chooser."""
        self.ls = ls
        self.matchup = MatchupModel(cfg, bridge)

    def _pair_score(self, my_sets, lead, opp_pair, off, dfn, spd):
        """One of my lead pairs against one opponent lead pair."""
        ls = self.ls
        offense = sum(max(off[i][k] for k in opp_pair) for i in lead)
        danger = sum(max(dfn[i][k] for k in opp_pair) for i in lead)
        speed = sum(spd[i][k] for i in lead for k in opp_pair) / \
            (len(lead) * len(opp_pair))
        syn = 0.0
        if any(self.matchup.has_move(my_sets[i], FAKEOUT_MOVES) for i in lead):
            syn += ls.w_syn_fakeout
        if any(self.matchup.has_move(my_sets[i], REDIRECT_MOVES) for i in lead):
            syn += ls.w_syn_redirect
        if speed < 0.4 and any(
                self.matchup.has_move(my_sets[i], SPEED_CONTROL_MOVES)
                for i in lead):
            syn += ls.w_syn_speedctl
        return offense - danger + ls.w_speed * (speed - 0.5) + syn

    def opp_lead_weights(self, n_opp, off, dfn):
        """Softmax weights over the opponent's lead pairs, best-for-them first.

        Their offense into me is my ``dfn`` and their danger is my ``off``,
        so their pair quality needs no second set of calc calls."""
        import math
        pairs = list(combinations(range(n_opp), min(2, n_opp)))
        n_my = len(off)
        scores = []
        for T in pairs:
            o = sum(max(dfn[i][k] for i in range(n_my)) for k in T)
            d = sum(max(off[i][k] for i in range(n_my)) for k in T)
            scores.append(o - d)
        mx = max(scores)
        ws = [math.exp(self.ls.opp_lead_temp * (s - mx)) for s in scores]
        tot = sum(ws)
        return [(T, w / tot) for T, w in zip(pairs, ws)]

    def _coverage(self, bring, n_opp, off, dfn):
        """Bring-4 coverage: mean best answer per opponent mon + walled term."""
        ls = self.ls
        per_opp = [max(off[i][k] - dfn[i][k] for i in bring)
                   for k in range(n_opp)]
        dent = [max(off[i][k] for i in bring) for k in range(n_opp)]
        if not per_opp:
            return 0.0
        return (ls.cov_mean * sum(per_opp) / len(per_opp)
                + ls.cov_worst * min(dent))

    def score_combos(self, my_sets, belief, n_bring=None):
        """Score every preview combo; return rows sorted best-first.

        Each row is ``(score, lead, back)``. Shared by the value selector
        (pruning) and by choose()."""
        n_bring = n_bring or self.ls.n_bring
        off, dfn, spd = self.matchup.tables(my_sets, belief)
        n_opp = len(belief.species)
        opp_ws = self.opp_lead_weights(n_opp, off, dfn)
        rows = []
        cov_cache = {}
        for lead, back in enumerate_previews(len(my_sets), n_bring):
            bring = tuple(sorted(lead + back))
            if bring not in cov_cache:
                cov_cache[bring] = self._coverage(bring, n_opp, off, dfn)
            s = sum(w * self._pair_score(my_sets, lead, T, off, dfn, spd)
                    for T, w in opp_ws)
            rows.append((s + self.ls.w_coverage * cov_cache[bring],
                         lead, back))
        rows.sort(key=lambda r: (-r[0], r[1], r[2]))
        return rows

    def choose(self, tracker, belief, my_id, n_bring=None):
        """Return ``(order, info)``: brought preview indices, leads first."""
        my_sets = [m.set for m in tracker.sides[my_id].mons]
        rows = self.score_combos(my_sets, belief, n_bring)
        s, lead, back = rows[0]
        info = {"kind": "expert", "score": s,
                "top": [(f"lead {list(L)} back {list(B)}", sc)
                        for sc, L, B in rows[:5]]}
        return preview_order(lead, back), info


class ExpertSwitchSelector:
    """Damage-calc replacement chooser for forced-switch requests."""

    def __init__(self, cfg, bridge=None, ls=LSCFG):
        """Build the matchup model; ``bridge`` may be shared with the chooser."""
        self.ls = ls
        self.matchup = MatchupModel(cfg, bridge)

    def _candidate_score(self, my_set, hp, boosts_status_mine, opp_alive,
                         belief, summary, fields, opp_extra):
        """Score one replacement against every live opponent active."""
        ls = self.ls
        opp_sets = [self.matchup._opp_particle(belief, k) for k in opp_alive]
        if not opp_sets:
            return 0.0
        out = self.matchup._bridge_fracs(
            [my_set], opp_sets, fields["off"], atk_extra={0: boosts_status_mine},
            dfd_extra=opp_extra)[0]
        inc = self.matchup._bridge_fracs(
            opp_sets, [my_set], fields["dfn"], atk_extra=opp_extra,
            dfd_extra={0: boosts_status_mine})
        mine_spe = self.matchup._speed(my_set)
        score = 0.0
        for j, k in enumerate(opp_alive):
            danger = inc[j][0] / max(hp, ls.sw_hp_floor)
            b = summary.get(k)
            if b is None or b["spe_hi"] <= b["spe_lo"]:
                p_fast = 0.5
            else:
                p_fast = min(1.0, max(0.0, (mine_spe - b["spe_lo"])
                                      / (b["spe_hi"] - b["spe_lo"])))
            score += (out[j] - ls.sw_in_weight * danger
                      + ls.sw_spd_weight * (p_fast - 0.5))
        return score / len(opp_alive)

    def choose(self, request, tracker, belief, my_id):
        """Return one Showdown choice string for a forceSwitch request."""
        opp_id = "p2" if my_id == "p1" else "p1"
        me, opp = tracker.sides[my_id], tracker.sides[opp_id]
        name_to_idx = {m.set["name"]: k for k, m in enumerate(me.mons)}
        idx_of_pos, _ = _pos_maps(request, name_to_idx)

        view = tracker._view(my_id)
        fields = {
            "off": {"weather": tracker.weather, "terrain": tracker.terrain,
                    "screens": [c for c, v in opp.conditions.items() if v and c
                                in ("reflect", "lightscreen", "auroraveil")]},
            "dfn": {"weather": tracker.weather, "terrain": tracker.terrain,
                    "screens": [c for c, v in me.conditions.items() if v and c
                                in ("reflect", "lightscreen", "auroraveil")]}}
        opp_alive = [m.team_idx for m in opp.mons
                     if m.active_slot is not None and not m.fainted]
        opp_extra = {}
        for j, k in enumerate(opp_alive):
            m = opp.mons[k]
            opp_extra[j] = {"boosts": {b: v for b, v in m.boosts.items()
                                       if b in ("atk", "def", "spa", "spd",
                                                "spe") and v},
                            "status": m.status if m.status == "brn" else ""}
        summary = belief.summary()

        scored = {}    # party position -> score
        for pos, p in enumerate(request["side"]["pokemon"], start=1):
            if p["active"] or p["condition"] == "0 fnt":
                continue
            try:
                idx = idx_of_pos(pos)
            except KeyError:
                continue
            mon = me.mons[idx]
            mine_extra = {"boosts": {}, "status":
                          mon.status if mon.status == "brn" else ""}
            scored[pos] = self._candidate_score(
                mon.set, mon.hp, mine_extra, opp_alive, belief, summary,
                fields, opp_extra)

        picks, out = set(), []
        for slot, force in enumerate(request.get("forceSwitch") or []):
            if not force:
                out.append("pass")
                continue
            options = [pos for pos in scored if pos not in picks]
            if not options:
                out.append("pass")
                continue
            best = max(options, key=lambda pos: (scored[pos], -pos))
            picks.add(best)
            out.append(f"switch {best}")
        return ", ".join(out) if out else "pass"
