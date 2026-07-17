"""Damage-calc matchup engine shared by every lead/switch-in selector.

The whole experiment (exp/lead-switch) needs one primitive: "how does my mon i
fare into their mon k, given what I currently believe about their set?" This
module answers it with the same `@smogon/calc` bridge the rest of the bot
uses (exact formulas, cached), and degrades to a type-chart estimate when no
bridge is available (unit tests, missing usage data).

Outputs are three [n_my, n_opp] float arrays:
  off[i, k]  best single-move expected damage fraction my i deals to their k
             (their defensive build = the belief filter's top particle)
  dfn[i, k]  best expected damage fraction their k deals to my i
             (their offensive moves = top particle's moves, or synthetic
             90 BP STABs when the prior has nothing)
  spd[i, k]  P(my i outspeeds their k), from my exact speed stat against the
             belief summary's effective-speed interval (which already absorbs
             scarf/nature/SP uncertainty)

This is deliberately the Game Freak trainer-AI shape (documented from the
Gen IV decompilation: scan for super-effective coverage, then rank by damage
calcs) rather than a learned model — it is the expert-system yardstick the
learned variants are measured against.
"""

from beliefs import calc_stat, load_dex
from config import CFG
from damage import request

# Gen 9 type effectiveness (attacker type -> {defender type: multiplier for
# the non-1.0 entries}). Static game knowledge; the sim's dex.json carries
# species/move types but not the chart itself.
TYPE_CHART = {
    "Normal": {"Rock": .5, "Ghost": 0, "Steel": .5},
    "Fire": {"Fire": .5, "Water": .5, "Grass": 2, "Ice": 2, "Bug": 2,
             "Rock": .5, "Dragon": .5, "Steel": 2},
    "Water": {"Fire": 2, "Water": .5, "Grass": .5, "Ground": 2, "Rock": 2,
              "Dragon": .5},
    "Electric": {"Water": 2, "Electric": .5, "Grass": .5, "Ground": 0,
                 "Flying": 2, "Dragon": .5},
    "Grass": {"Fire": .5, "Water": 2, "Grass": .5, "Poison": .5, "Ground": 2,
              "Flying": .5, "Bug": .5, "Rock": 2, "Dragon": .5, "Steel": .5},
    "Ice": {"Fire": .5, "Water": .5, "Grass": 2, "Ice": .5, "Ground": 2,
            "Flying": 2, "Dragon": 2, "Steel": .5},
    "Fighting": {"Normal": 2, "Ice": 2, "Poison": .5, "Flying": .5,
                 "Psychic": .5, "Bug": .5, "Rock": 2, "Ghost": 0, "Dark": 2,
                 "Steel": 2, "Fairy": .5},
    "Poison": {"Grass": 2, "Poison": .5, "Ground": .5, "Rock": .5, "Ghost": .5,
               "Steel": 0, "Fairy": 2},
    "Ground": {"Fire": 2, "Electric": 2, "Grass": .5, "Poison": 2,
               "Flying": 0, "Bug": .5, "Rock": 2, "Steel": 2},
    "Flying": {"Electric": .5, "Grass": 2, "Fighting": 2, "Bug": 2,
               "Rock": .5, "Steel": .5},
    "Psychic": {"Fighting": 2, "Poison": 2, "Psychic": .5, "Dark": 0,
                "Steel": .5},
    "Bug": {"Fire": .5, "Grass": 2, "Fighting": .5, "Poison": .5, "Flying": .5,
            "Psychic": 2, "Ghost": .5, "Dark": 2, "Steel": .5, "Fairy": .5},
    "Rock": {"Fire": 2, "Ice": 2, "Fighting": .5, "Ground": .5, "Flying": 2,
             "Bug": 2, "Steel": .5},
    "Ghost": {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": .5},
    "Dragon": {"Dragon": 2, "Steel": .5, "Fairy": 0},
    "Dark": {"Fighting": .5, "Psychic": 2, "Ghost": 2, "Dark": .5,
             "Fairy": .5},
    "Steel": {"Fire": .5, "Water": .5, "Electric": .5, "Ice": 2, "Rock": 2,
              "Steel": .5, "Fairy": 2},
    "Fairy": {"Fire": .5, "Fighting": 2, "Poison": .5, "Dragon": 2,
              "Dark": 2, "Steel": .5},
}

# moves that matter to lead synergy scoring, by role
FAKEOUT_MOVES = {"fakeout"}
REDIRECT_MOVES = {"ragepowder", "followme"}
SPEED_CONTROL_MOVES = {"tailwind", "trickroom", "icywind", "electroweb",
                       "bleakwindstorm", "stringshot"}
# moves that are never a mon's "best damage" (status etc. carry basePower 0
# in the dex, so the damage ranking already ignores them)


def type_multiplier(move_type, defender_types):
    """Product of ``TYPE_CHART`` multipliers of one move type into 1-2 types."""
    out = 1.0
    for t in defender_types:
        out *= TYPE_CHART.get(move_type, {}).get(t, 1.0)
    return out


class MatchupModel:
    """Belief-aware pairwise matchup tables from the damage calculator."""

    def __init__(self, cfg=CFG, bridge=None):
        """Cache the dex; keep a (possibly None) shared ``DamageBridge``."""
        self.cfg = cfg
        self.bridge = bridge
        self.dex = load_dex(cfg) or {"species": {}, "moves": {}, "items": {}}

    # -- set plumbing -------------------------------------------------------
    def _species_types(self, species_sid):
        """Types list for a species sid, or empty when the dex lacks it."""
        sp = self.dex["species"].get(species_sid)
        return sp["types"] if sp else []

    def _opp_particle(self, belief, k):
        """The belief filter's modal set for opponent preview index ``k``."""
        p = belief.top_particle(k)
        return {"species": belief.species[k], "moves": list(p["moves"]),
                "item": p["item"], "ability": p["ability"],
                "nature": p["nature"], "evs": p.get("evs"), "level": 50}

    def _attack_moves(self, set_):
        """The set's damaging move sids (dex category != Status)."""
        out = []
        for mv in set_.get("moves", ()):
            d = self.dex["moves"].get(mv)
            if d and d["category"] != "Status" and d["basePower"] > 0:
                out.append(mv)
        return out

    def _synthetic_stabs(self, species_sid):
        """Placeholder 90 BP STAB ids when a set's moves are unknown.

        The bridge would reject fake move names, so callers only use these on
        the type-chart path; they exist to keep an uncovered species from
        scoring as harmless."""
        return [("__stab__", t) for t in self._species_types(species_sid)]

    def _speed(self, set_):
        """Exact speed stat of a fully known set (my team side)."""
        base = self.dex["species"].get(_sid(set_["species"]), {}).get(
            "baseStats", {}).get("spe")
        if base is None:
            return 100.0
        evs = set_.get("evs") or [0] * 6
        spe = calc_stat(base, "spe", set_.get("nature", "serious"), evs[5])
        if set_.get("item") == "choicescarf":
            spe *= 1.5
        return spe

    # -- damage primitives ---------------------------------------------------
    def _chart_frac(self, atk_set, dfd_species_sid):
        """Type-chart damage estimate (no bridge): best move's
        BP x effectiveness x STAB, scaled so a neutral 90 BP hit ~= 0.35."""
        dfd_types = self._species_types(dfd_species_sid)
        atk_types = self._species_types(_sid(atk_set["species"]))
        best = 0.0
        moves = self._attack_moves(atk_set)
        if moves:
            for mv in moves:
                d = self.dex["moves"][mv]
                eff = type_multiplier(d["type"], dfd_types)
                stab = 1.5 if d["type"] in atk_types else 1.0
                best = max(best, d["basePower"] * eff * stab)
        else:
            for _, t in self._synthetic_stabs(_sid(atk_set["species"])):
                best = max(best, 90 * type_multiplier(t, dfd_types) * 1.5)
        return min(1.5, best * 0.35 / (90 * 1.5))

    def _bridge_fracs(self, atk_sets, dfd_sets, field=None,
                      atk_extra=None, dfd_extra=None):
        """Batched best-move expected damage fraction for every (atk, dfd).

        atk_sets/dfd_sets: full set dicts. atk_extra/dfd_extra: optional
        per-index dicts merged into the calc request (boosts, status,
        alliesFainted). Returns [len(atk)][len(dfd)] floats; falls back to the
        type chart per-pair when the bridge or a calc result is missing."""
        n_a, n_d = len(atk_sets), len(dfd_sets)
        out = [[0.0] * n_d for _ in range(n_a)]
        keys, reqs = [], []
        for i, a in enumerate(atk_sets):
            moves = self._attack_moves(a)
            atk = {"species": _sid(a["species"]), "level": a.get("level", 50),
                   "item": a.get("item"), "ability": a.get("ability"),
                   "nature": a.get("nature"), "evs": a.get("evs"),
                   **((atk_extra or {}).get(i) or {})}
            for k, d in enumerate(dfd_sets):
                dfd = {"species": _sid(d["species"]),
                       "level": d.get("level", 50), "item": d.get("item"),
                       "ability": d.get("ability"), "nature": d.get("nature"),
                       "evs": d.get("evs"),
                       **((dfd_extra or {}).get(k) or {})}
                for mv in moves:
                    keys.append((i, k))
                    reqs.append(request(atk, dfd, mv, field))
        calced = set()
        if self.bridge and reqs:
            for (i, k), r in zip(keys, self.bridge.calc_batch(reqs)):
                if r is not None:
                    calced.add((i, k))
                    out[i][k] = max(out[i][k], (r[0] + r[1]) / 2)
        for i, a in enumerate(atk_sets):    # no bridge / failed calc / no moves
            for k, d in enumerate(dfd_sets):
                if (i, k) not in calced:
                    out[i][k] = max(out[i][k],
                                    self._chart_frac(a, _sid(d["species"])))
        return out

    # -- the tables -----------------------------------------------------------
    def tables(self, my_sets, belief, field=None, my_extra=None,
               opp_extra=None):
        """Return ``(off, dfn, spd)`` matchup tables (module docstring).

        my_sets: my full set dicts, team-preview order. belief: the live
        ``OpponentBelief`` (its species list fixes the opponent axis). field:
        optional shared calc field dict. my_extra/opp_extra: per-index request
        overrides such as current boosts or a burn."""
        opp_sets = [self._opp_particle(belief, k)
                    for k in range(len(belief.species))]
        off = self._bridge_fracs(my_sets, opp_sets, field,
                                 atk_extra=my_extra, dfd_extra=opp_extra)
        dfn_t = self._bridge_fracs(opp_sets, my_sets, field,
                                   atk_extra=opp_extra, dfd_extra=my_extra)
        dfn = [[dfn_t[k][i] for k in range(len(opp_sets))]
               for i in range(len(my_sets))]
        summary = belief.summary()
        spd = []
        for s in my_sets:
            mine = self._speed(s)
            row = []
            for k in range(len(belief.species)):
                b = summary.get(k)
                if b is None:
                    row.append(0.5)
                    continue
                lo, hi = b["spe_lo"], b["spe_hi"]
                if hi <= lo:
                    row.append(1.0 if mine > lo else 0.0)
                else:
                    row.append(min(1.0, max(0.0, (mine - lo) / (hi - lo))))
            spd.append(row)
        return off, dfn, spd

    def has_move(self, set_, move_ids):
        """True when the set carries any move in ``move_ids``."""
        return bool(set(set_.get("moves", ())) & set(move_ids))


def _sid(name):
    """Lowercase alphanumeric Showdown id (same rule as data.sid)."""
    import re
    return re.sub(r"[^a-z0-9]", "", str(name).lower())
